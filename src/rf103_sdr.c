/*
 * rf103_test - simple stream test program  for librf103
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "rf103.h"
#include "wavewrite.h"


static void count_bytes_callback(uint32_t data_size, uint8_t *data,
                                 void *context);

static int stop_reception = 0;

int main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: %s <image file> <sample rate>\n", argv[0]);
    return -1;
  }
  char *imagefile = argv[1];
  double sample_rate = 0.0;
  sscanf(argv[2], "%lf", &sample_rate);

  if (sample_rate <= 0) {
    fprintf(stderr, "ERROR - given samplerate '%f' should be > 0\n", sample_rate);
    return -1;
  }

  int ret_val = -1;

  rf103_t *rf103 = rf103_open(0, imagefile);
  if (rf103 == 0) {
    fprintf(stderr, "ERROR - rf103_open() failed\n");
    return -1;
  }

  if (rf103_set_sample_rate(rf103, sample_rate) < 0) {
    fprintf(stderr, "ERROR - rf103_set_sample_rate() failed\n");
    goto DONE;
  }

  if (rf103_set_async_params(rf103, 0, 0, count_bytes_callback, rf103) < 0) {
    fprintf(stderr, "ERROR - rf103_set_async_params() failed\n");
    goto DONE;
  }

  if (rf103_start_streaming(rf103) < 0) {
    fprintf(stderr, "ERROR - rf103_start_streaming() failed\n");
    return -1;
  }

  fprintf(stderr, "started streaming...\n");

  /* todo: move this into a thread */
  stop_reception = 0;
  while (!stop_reception)
    rf103_handle_events(rf103);

  fprintf(stderr, "finished. now stop streaming ..\n");
  if (rf103_stop_streaming(rf103) < 0) {
    fprintf(stderr, "ERROR - rf103_stop_streaming() failed\n");
    return -1;
  }
  /* done - all good */
  ret_val = 0;

DONE:
  rf103_close(rf103);

  return ret_val;
}

static void count_bytes_callback(uint32_t data_size,
                                 uint8_t *data,
                                 void *context __attribute__((unused)) )
{
  if (stop_reception)
    return;
  fwrite(data, sizeof(uint8_t), data_size, stdout);
}

