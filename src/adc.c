/*
 * adc.c - ADC related functions (including bulk data transfer)
 *
 * Copyright (C) 2020 by Franco Venturi
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* References:
 *  - librtlsdr.c: https://github.com/librtlsdr/librtlsdr/blob/development/src/librtlsdr.c
 *  - Ettus Research UHD libusb1_zero_copy.cpp: https://github.com/EttusResearch/uhd/blob/master/host/lib/transport/libusb1_zero_copy.cpp
 */

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdatomic.h>

#include "adc.h"
#include "usb_device.h"
#include "usb_device_internals.h"
#include "logging.h"


typedef struct adc adc_t;

/* internal functions */
static void adc_read_async_callback(struct libusb_transfer *transfer);


enum ADCStatus {
  ADC_STATUS_OFF,
  ADC_STATUS_READY,
  ADC_STATUS_STREAMING,
  ADC_STATUS_CANCELLED,
  ADC_STATUS_FAILED = 0xff
};

typedef struct adc {
  enum ADCStatus status;
  int random;
  usb_device_t *usb_device;
  uint32_t sample_rate;
  uint32_t frame_size;
  uint32_t num_frames;
  rf103_read_async_cb_t callback;
  void *callback_context;
  uint8_t **frames;
  struct libusb_transfer **transfers;
  atomic_int active_transfers;
} adc_t;


static const uint32_t DEFAULT_ADC_SAMPLE_RATE = 64000000;   /* 64Msps */
static const uint32_t DEFAULT_ADC_FRAME_SIZE = (2 * DEFAULT_ADC_SAMPLE_RATE / 1000);  /* ~ 1 ms */
static const uint32_t DEFAULT_ADC_NUM_FRAMES = 96;  /* we should not exceed 120 ms in total! */
const unsigned int BULK_XFER_TIMEOUT = 5000; // timeout (in ms) for each bulk transfer


adc_t *adc_open_sync(usb_device_t *usb_device)
{
  adc_t *ret_val = 0;

  /* we must have a bulk in device to transfer data from */
  if (usb_device->bulk_in_endpoint_address == 0) {
    log_error("no USB bulk in endpoint found", __func__, __FILE__, __LINE__);
    return ret_val;
  }

  /* we are good here - create and initialize the adc */
  adc_t *this = (adc_t *) malloc(sizeof(adc_t));
  this->status = ADC_STATUS_READY;
  this->random = 0;
  this->usb_device = usb_device;
  this->sample_rate = DEFAULT_ADC_SAMPLE_RATE;
  this->frame_size = 0;
  this->num_frames = 0;
  this->callback = 0;
  this->callback_context = 0;
  this->frames = 0;
  this->transfers = 0;
  atomic_init(&this->active_transfers, 0);

  ret_val = this;
  return ret_val;
}


adc_t *adc_open_async(usb_device_t *usb_device, uint32_t frame_size,
                      uint32_t num_frames, rf103_read_async_cb_t callback,
                      void *callback_context)
{
  adc_t *ret_val = 0;

  /* we must have a bulk in device to transfer data from */
  if (usb_device->bulk_in_endpoint_address == 0) {
    log_error("no USB bulk in endpoint found", __func__, __FILE__, __LINE__);
    return ret_val;
  }

  /* frame size must be a multiple of max_packet_size * max_burst */
  uint32_t max_xfer_size = usb_device->bulk_in_max_packet_size *
                           usb_device->bulk_in_max_burst;
  if ( !max_xfer_size ) {
    fprintf(stderr, "ERROR: maximum transfer size is 0. probably not connected at USB 3 port?!\n");
    return ret_val;
  }

  num_frames = num_frames > 0 ? num_frames : DEFAULT_ADC_NUM_FRAMES;
  frame_size = frame_size > 0 ? frame_size : DEFAULT_ADC_FRAME_SIZE;
  frame_size = max_xfer_size * ((frame_size +max_xfer_size -1) / max_xfer_size);  // round up
  int iso_packets_per_frame = frame_size / usb_device->bulk_in_max_packet_size;
  fprintf(stderr, "frame_size = %u, iso_packets_per_frame = %d\n", (unsigned)frame_size, iso_packets_per_frame);

  if (frame_size % max_xfer_size != 0) {
    fprintf(stderr, "ADC frame size must be a multiple of %d\n", max_xfer_size);
    return ret_val;
  }

  /* allocate frames for zerocopy USB bulk transfers */
  uint8_t **frames = (uint8_t **) malloc(num_frames * sizeof(uint8_t *));
  for (uint32_t i = 0; i < num_frames; ++i) {
    frames[i] = libusb_dev_mem_alloc(usb_device->dev_handle, frame_size);
    if (frames[i] == 0) {
      log_error("libusb_dev_mem_alloc() failed", __func__, __FILE__, __LINE__);
      for (uint32_t j = 0; j < i; j++) {
        libusb_dev_mem_free(usb_device->dev_handle, frames[j], frame_size);
      }
      return ret_val;
    }
  }

  /* we are good here - create and initialize the adc */
  adc_t *this = (adc_t *) malloc(sizeof(adc_t));
  this->status = ADC_STATUS_READY;
  this->random = 0;
  this->usb_device = usb_device;
  this->sample_rate = DEFAULT_ADC_SAMPLE_RATE;
  this->frame_size = frame_size > 0 ? frame_size : DEFAULT_ADC_FRAME_SIZE;
  this->num_frames = num_frames > 0 ? num_frames : DEFAULT_ADC_NUM_FRAMES;
  this->callback = callback;
  this->callback_context = callback_context;
  this->frames = frames;

  /* populate the required libusb_transfer fields */
  struct libusb_transfer **transfers = (struct libusb_transfer **) malloc(num_frames * sizeof(struct libusb_transfer *));
  for (uint32_t i = 0; i < num_frames; ++i) {
    transfers[i] = libusb_alloc_transfer(0);	// iso_packets_per_frame ?
    libusb_fill_bulk_transfer(transfers[i], usb_device->dev_handle,
                              usb_device->bulk_in_endpoint_address,
                              frames[i], frame_size, adc_read_async_callback,
                              this, BULK_XFER_TIMEOUT);
  }
  this->transfers = transfers;
  atomic_init(&this->active_transfers, 0);

  ret_val = this;
  return ret_val;
}


void adc_close(adc_t *this)
{
  if (this->transfers) {
    for (uint32_t i = 0; i < this->num_frames; ++i) {
      libusb_free_transfer(this->transfers[i]);
    }
    free(this->transfers);
  }
  if (this->frames != 0) {
    for (uint32_t i = 0; i < this->num_frames; ++i) {
      libusb_dev_mem_free(this->usb_device->dev_handle, this->frames[i],
                          this->frame_size);
    }
    free(this->frames);
  }
  free(this);
  return;
}


int adc_set_random(adc_t *this, int random)
{
  this->random = random;
  return 0;
}


int adc_set_sample_rate(adc_t *this, uint32_t sample_rate)
{
  /* no checks yet */
  this->sample_rate = sample_rate;
  return 0;
}


int adc_start(adc_t *this)
{
  if (this->status != ADC_STATUS_READY) {
    fprintf(stderr, "ERROR - adc_start() called with ADC status not READY: %d\n", this->status);
    return -1;
  }

  /* if there is no callback, then streaming is synchronous - nothing to do */
  if (this->callback == 0) {
    this->status = ADC_STATUS_STREAMING;
    return 0;
  }

  /* submit all the transfers */
  atomic_init(&this->active_transfers, 0);
  for (uint32_t i = 0; i < this->num_frames; ++i) {
    int ret = libusb_submit_transfer(this->transfers[i]);
    if (ret < 0) {
      log_usb_error(ret, __func__, __FILE__, __LINE__);
      this->status = ADC_STATUS_FAILED;
      return -1;
    }
    atomic_fetch_add(&this->active_transfers, 1);
  }

  this->status = ADC_STATUS_STREAMING;

  return 0;
}


int adc_stop(adc_t *this)
{
  /* if there is no callback, then streaming is synchronous - nothing to do */
  if (this->callback == 0) {
    if (this->status == ADC_STATUS_STREAMING) {
      this->status = ADC_STATUS_READY;
    }
    return 0;
  }

  this->status = ADC_STATUS_CANCELLED;
  /* cancel all the active transfers */
  for (uint32_t i = 0; i < this->num_frames; ++i) {
    int ret = libusb_cancel_transfer(this->transfers[i]);
    if (ret < 0) {
      if (ret == LIBUSB_ERROR_NOT_FOUND) {
        continue;
      }
      log_usb_error(ret, __func__, __FILE__, __LINE__);
      this->status = ADC_STATUS_FAILED;
    }
  }

  /* flush all the events */
  struct timeval noblock = { 0, 0 };
  int ret = libusb_handle_events_timeout_completed(0, &noblock, 0);
  if (ret < 0) {
    log_usb_error(ret, __func__, __FILE__, __LINE__);
    this->status = ADC_STATUS_FAILED;
  }

  return 0;
}


int adc_reset_status(adc_t *this)
{
  switch (this->status) {
    case ADC_STATUS_READY:
      /* nothing to do here */
      return 0;
    case ADC_STATUS_CANCELLED:
    case ADC_STATUS_FAILED:
      if (this->active_transfers > 0) {
        fprintf(stderr, "ERROR - adc_reset_status() called with %d transfers still active\n",
                        this->active_transfers);
        return -1;
      }
      break;
    default:
      fprintf(stderr, "ERROR - adc_reset_status() called with invalid status: %d\n",
                      this->status);
      return -1;
  }

  /* we are good here; reset the status */
  this->status = ADC_STATUS_READY;
  return 0;
}


int adc_read_sync(adc_t *this, uint8_t *data, int length, int *transferred)
{
  int ret = libusb_bulk_transfer(this->usb_device->dev_handle,
                                 this->usb_device->bulk_in_endpoint_address,
                                 data, length, transferred, BULK_XFER_TIMEOUT);
  if (ret < 0) {
    log_usb_error(ret, __func__, __FILE__, __LINE__);
    return -1;
  }

  /* remove ADC randomization */
  if (this->random) {
    uint16_t *samples = (uint16_t *) data;
    int n = *transferred / 2;
    for (int i = 0; i < n; ++i) {
      if (samples[i] & 1) {
        samples[i] ^= 0xfffe;
      }
    }
  }

  return 0;
}


/* internal functions */
static void LIBUSB_CALL adc_read_async_callback(struct libusb_transfer *transfer)
{
  adc_t *this = (adc_t *) transfer->user_data;
  int ret;
  switch (transfer->status) {
    case LIBUSB_TRANSFER_COMPLETED:
      /* success!!! */
      if (this->status == ADC_STATUS_STREAMING) {
        /* remove ADC randomization */
        if (this->random) {
          uint16_t *samples = (uint16_t *) transfer->buffer;
          int n = transfer->actual_length / 2;
          for (int i = 0; i < n; ++i) {
            if (samples[i] & 1) {
              samples[i] ^= 0xfffe;
            }
          }
        }
        this->callback(transfer->actual_length, transfer->buffer,
                       this->callback_context);
        ret = libusb_submit_transfer(transfer);
        if (ret == 0) {
          return;
        }
        log_usb_error(ret, __func__, __FILE__, __LINE__);
      }
      break;
    case LIBUSB_TRANSFER_CANCELLED:
      /* librtlsdr does also ignore LIBUSB_TRANSFER_CANCELLED */
      return;
    case LIBUSB_TRANSFER_ERROR:
    case LIBUSB_TRANSFER_TIMED_OUT:
    case LIBUSB_TRANSFER_STALL:
    case LIBUSB_TRANSFER_NO_DEVICE:
    case LIBUSB_TRANSFER_OVERFLOW:
      log_usb_error(transfer->status, __func__, __FILE__, __LINE__);
      break;
  }

  this->status = ADC_STATUS_FAILED;
  atomic_fetch_sub(&this->active_transfers, 1);
  fprintf(stderr, "Cancelling\n");
  /* cancel all the active transfers */
  for (uint32_t i = 0; i < this->num_frames; ++i) {
    int ret = libusb_cancel_transfer(transfer);
    if (ret < 0) {
      if (ret == LIBUSB_ERROR_NOT_FOUND) {
        continue;
      }
      log_usb_error(ret, __func__, __FILE__, __LINE__);
    }
  }
  return;
}

