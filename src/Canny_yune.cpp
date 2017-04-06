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

#include "Canny_yune.h"
#include <math.h>
#include <opencv2\opencv.hpp>
#include <Windows.h>

using namespace std;
using namespace cv;

void Canny_yune(const Mat1b &src, int th_high, int th_low, Mat1b &Edgeimg) {
	int h = src.rows;
	int w = src.cols;

	int i, j;

	int dx, dy, mag, direction;
	double slope;
	int index, index2;

	const double tan225 = 0.4142;
	const double tan675 = 2.4142;

	const int STRONG_EDGE = 255;
	const int WEAK_EDGE = 128;

	int *magimg = (int *)fastMalloc(sizeof(int)*(w*h));
	int *dx_grad = (int *)fastMalloc(sizeof(int)*(w*h));
	int *dy_grad = (int *)fastMalloc(sizeof(int)*(w*h));
	BYTE * Edge = Edgeimg.reshape(1, 1).ptr<uchar>(0);
	BYTE **stack_top, **stack_bottom;

	stack_top = new BYTE*[w*h];
	stack_bottom = stack_top;

	//sobel mask
	for (i = 1; i < h - 1; i++) {
		index = i*w;
		const uchar* const img_row = src.ptr<uchar>(i);
		const uchar* const img_row_prev = (uchar *)(((char *)img_row) - src.step.p[0]);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + src.step.p[0]);
		for (j = 1; j < w - 1; j++) {
			index2 = index + j;
			dx = img_row_prev[j + 1] + 2 * img_row[j + 1] + img_row_fol[j + 1]
				- img_row_prev[j - 1] - 2 * img_row[j - 1] - img_row_fol[j - 1];
			dy = -img_row_prev[j - 1] - 2 * img_row_prev[j] - img_row_prev[j + 1]
				+ img_row_fol[j - 1] + 2 * img_row_fol[j] + img_row_fol[j + 1];
			mag = abs(dx) + abs(dy);
			dx_grad[index2] = dx;
			dy_grad[index2] = dy;
			magimg[index2] = mag;
		}
	}
	//non-maximum suppression
	bool edge_canditate;
	for (i = 1; i < h - 1; i++) {
		index = i*w;
		for (j = 1; j < w - 1; j++) {
			index2 = index + j;
			mag = magimg[index2];
			if (mag > th_low) {
				dx = dx_grad[index2];
				dy = dy_grad[index2];
				if (dx != 0) {
					slope = double(dy) / double(dx);
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

				edge_canditate = true;
				//non-maxima suppression
				if (direction == 0) {
					if (mag < magimg[index2 - 1] || mag < magimg[index2 + 1])
						edge_canditate = false;
				}
				else if (direction == 1) {
					if (mag < magimg[index2 + w + 1] || mag < magimg[index2 - w - 1])
						edge_canditate = false;
				}
				else if (direction == 2) {
					if (mag < magimg[index2 + w] || mag < magimg[index2 - w])
						edge_canditate = false;
				}
				else {
					if (mag < magimg[index2 + w - 1] || mag < magimg[index2 - w + 1])
						edge_canditate = false;
				}
				if (edge_canditate) {
					if (mag > th_high) {
						Edge[index2] = STRONG_EDGE;
						*(stack_top++) = (BYTE*)(Edge + index2);
					}
					else
						Edge[index2] = WEAK_EDGE;
				}
			}
		}
	}

#define STRONG_PUSH(p) *(p) = STRONG_EDGE, *(stack_top++) = (p)
#define STRONG_POP()  *(--stack_top)
	//Include weak edges which is neighborhood of strong edge
	while (stack_top != stack_bottom) {
		BYTE* p = STRONG_POP();

		if (*(p + 1) == WEAK_EDGE)
			STRONG_PUSH(p + 1);
		if (*(p - 1) == WEAK_EDGE)
			STRONG_PUSH(p - 1);
		if (*(p + w) == WEAK_EDGE)
			STRONG_PUSH(p + w);
		if (*(p - w) == WEAK_EDGE)
			STRONG_PUSH(p - w);
		if (*(p - w - 1) == WEAK_EDGE)
			STRONG_PUSH(p - w - 1);
		if (*(p - w + 1) == WEAK_EDGE)
			STRONG_PUSH(p - w + 1);
		if (*(p + w - 1) == WEAK_EDGE)
			STRONG_PUSH(p + w - 1);
		if (*(p + w + 1) == WEAK_EDGE)
			STRONG_PUSH(p + w + 1);
	}
	for (i = 0; i < w*h; i++)
		if (Edge[i] != STRONG_EDGE)
			Edge[i] = 0;

	fastFree(magimg);
	fastFree(dx_grad);
	fastFree(dy_grad);
	delete[] stack_bottom;
}