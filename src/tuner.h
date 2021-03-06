/*
 * tuner.h - R820T2 functions
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

#ifndef __TUNER_H
#define __TUNER_H

#include "usb_device.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct tuner tuner_t;


int has_tuner(usb_device_t *usb_device);

tuner_t *tuner_open(usb_device_t *usb_device);

void tuner_close(tuner_t *this);

uint32_t tuner_get_xtal_frequency(tuner_t *this);

int tuner_set_xtal_frequency(tuner_t *this, uint32_t xtal_frequency);

uint32_t tuner_get_if_frequency(tuner_t *this);

int tuner_set_if_frequency(tuner_t *this, uint32_t if_frequency);

int tuner_set_frequency(tuner_t *this, double frequency);

int tuner_set_harmonic_frequency(tuner_t *this, double frequency, int harmonic);

int tuner_get_lna_gains(tuner_t *this, const int *gains[]);

int tuner_set_lna_gain(tuner_t *this, int gain);

int tuner_set_lna_agc(tuner_t *this, int agc);

int tuner_get_mixer_gains(tuner_t *this, const int *gains[]);

int tuner_set_mixer_gain(tuner_t *this, int gain);

int tuner_set_mixer_agc(tuner_t *this, int agc);

int tuner_get_vga_gains(tuner_t *this, const int *gains[]);

int tuner_set_vga_gain(tuner_t *this, int gain);

int tuner_get_if_bandwidths(tuner_t *this, uint32_t *if_bandwidths[]);

int tuner_set_if_bandwidth(tuner_t *this, uint32_t bandwidth);

int tuner_start(tuner_t *this);

int tuner_stop(tuner_t *this);

int tuner_standby(tuner_t *this);

#ifdef __cplusplus
}
#endif

#endif /* __TUNER_H */
