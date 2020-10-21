/*
 * rf103_sdr - simple stream test program  for librf103
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
#include <getopt.h>
#include <signal.h>

#include "rf103.h"


static void count_bytes_callback(uint32_t data_size, uint8_t *data,
                                 void *context);

static int stop_reception = 0;

void print_usage() {
  fprintf(stderr,
    "Usage: rf103_sdr [options]\n\n"
    "Available options:\n"
    " -h, --help              show this message\n"
    " -i, --imagefile         firmware image file\n"
    " -s, --samplerate        use the specified samplerate\n"
    " -a, --attenuation       set SDR attenuation (default: off)\n"
  );
}

static void sighandler(int signum) {
  fprintf(stderr, "Signal %i caught, exiting!\n", signum);
  stop_reception = 1;
}


int main(int argc, char **argv)
{
  double sample_rate = 0.0;
  double attenuation = 0;
  char *imagefile = NULL;
  struct sigaction sigact;

  static struct option long_options[] = {
    {"help", no_argument, NULL, 'h'},
    {"imagefile", required_argument, NULL, 'i'},
    {"samplerate", required_argument, NULL, 's'},
    {"attenuation", required_argument, NULL, 'a'},
    { NULL, 0, NULL, 0 }
  };

  int c;

  while ((c = getopt_long(argc, argv, "hi:f:s:a:", long_options, NULL)) != -1) {
    switch (c) {
      case 'h':
        print_usage();
        break;
      case 'i':
        imagefile = optarg;
        break;
      case 's':
        sample_rate = (double)strtoul(optarg, NULL, 10);
        break;
      case 'a':
        attenuation = (double)strtoul(optarg, NULL, 10);
        break;
    }
  }

  if (imagefile == NULL) {
    print_usage();
    return -1;
  }

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

  sigact.sa_handler = sighandler;
  sigemptyset(&sigact.sa_mask);
  sigact.sa_flags = 0;
  sigaction(SIGINT, &sigact, NULL);
  sigaction(SIGTERM, &sigact, NULL);
  sigaction(SIGQUIT, &sigact, NULL);
  sigaction(SIGPIPE, &sigact, NULL);

  if (rf103_set_sample_rate(rf103, sample_rate) < 0) {
    fprintf(stderr, "ERROR - rf103_set_sample_rate() failed\n");
    goto DONE;
  }

  if (rf103_hf_attenuation(rf103, attenuation) < 0) {
    fprintf(stderr, "ERROR - rf103_hf_attenuation() failed\n");
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

