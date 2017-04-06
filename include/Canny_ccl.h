// Copyright(c) 2017 Yunseok Jang, Junwon Mun, and Jaeseok Kim
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once
#include "opencv2/opencv.hpp"
#include "equivalenceSolverSuzuki.h"

inline void check_edge(const int& w, const int& index2, const int * mag_tbl, const int * dx_tbl, const int * dy_tbl, bool &bMaxima, const int &mag) {
	int dx, dy, slope, direction;
	const int fbit = 10;
	const int tan225 = 424; //tan25.5 << fbit, 0.4142
	const int tan675 = 2472; //tan67.5 << fbit, 2.4142

	const int CERTAIN_EDGE = 255;
	const int PROBABLE_EDGE = 100;
	dx = dx_tbl[index2];
	dy = dy_tbl[index2];
	if (dx != 0) {
		slope = (dy << fbit) / dx;
		if (slope > 0) {
			if (slope < tan225)
				direction = 0;
			else if (slope < tan675)
				direction = 1;
			else
				direction = 2;
		}
		else {
			if (-slope > tan675)
				direction = 2;
			else if (-slope > tan225)
				direction = 3;
			else
				direction = 0;
		}
	}
	else
		direction = 2;

	bMaxima = true;
	if (direction == 0) {
		if (mag < mag_tbl[index2 - 1] || mag < mag_tbl[index2 + 1])
			bMaxima = false;
	}
	else if (direction == 1) {
		if (mag < mag_tbl[index2 + w + 1] || mag < mag_tbl[index2 - w - 1])
			bMaxima = false;
	}
	else if (direction == 2) {
		if (mag < mag_tbl[index2 + w] || mag < mag_tbl[index2 - w])
			bMaxima = false;
	}
	else {
		if (mag < mag_tbl[index2 + w - 1] || mag < mag_tbl[index2 - w + 1])
			bMaxima = false;
	}

}

int Canny_ccl(const cv::Mat1b &mg, cv::Mat1i &imgLabels);