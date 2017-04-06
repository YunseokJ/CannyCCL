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

#include "Canny_ccl.h"

using namespace cv;
using namespace std;

int Canny_ccl(const Mat1b &img, Mat1i &imgLabels) {
	imgLabels = cv::Mat1i(img.size(), 0);
	const int th_high = 180;
	const int th_low = 70;

	int h = img.rows;
	int w = img.cols;
	int i, j;

	int dx, dy, mag, direction;
	double slope;
	int index, index2;

	const double tan225 = 0.4142;
	const double tan675 = 2.4142;
	
	bool bMaxima;
	int *mag_tbl = (int *)fastMalloc(sizeof(int)*(w*h));
	int *dx_tbl = (int *)fastMalloc(sizeof(int)*(w*h));
	int *dy_tbl = (int *)fastMalloc(sizeof(int)*(w*h));

	
	//Maximum number of labels
	const size_t Plength = h*w / 4;
	//Tree of labels
	uint *P = (uint *)fastMalloc(sizeof(uint)* Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;
	// strong edge vector
	bool *sev = (bool *)fastMalloc(sizeof(bool)* Plength);
	sev[0] = 0;


	//sobel Edge detection
	for (i = 1; i < h - 1; i++) {
		index = i*w;
		const uchar* const img_row = img.ptr<uchar>(i);
		const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		for (j = 1; j < w - 1; j++) {
			index2 = index + j;
			dx = img_row_prev[j + 1] + 2 * img_row[j + 1] + img_row_fol[j + 1]
				- img_row_prev[j - 1] - 2 * img_row[j - 1] - img_row_fol[j - 1];
			dy = -img_row_prev[j - 1] - 2 * img_row_prev[j] - img_row_prev[j + 1]
				+ img_row_fol[j - 1] + 2 * img_row_fol[j] + img_row_fol[j + 1];
			mag = abs(dx) + abs(dy);
			dx_tbl[index2] = dx;
			dy_tbl[index2] = dy;
			mag_tbl[index2] = mag;
		}
	}

#define condition_p imgLabels_prev[j-1]>0
#define condition_q imgLabels_prev[j]>0
#define condition_r imgLabels_prev[j+1]>0

	for (i = 1; i < h - 1; i++) {
		uint* const imgLabels_row = imgLabels.ptr<uint>(i);
		uint* const imgLabels_prev = (uint *)(((char*)imgLabels_row) - imgLabels.step.p[0]);
		index = i*w;
		// Second col
		j = 1; index2 = index + j;
		mag = mag_tbl[index2];
		if (mag > th_low) {
			dx = dx_tbl[index2];
			dy = dy_tbl[index2];
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

			bMaxima = true;
			//non-maxima suppression
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
			if (bMaxima) {
				if (mag > th_high) {
					//strong edge
					if (condition_q) {
						imgLabels_row[j] = imgLabels_prev[j];
						sev[findRoot(P, imgLabels_prev[j])] = 1;
						goto tree_B;
					}
					else {
						if (condition_r) {
							imgLabels_row[j] = imgLabels_prev[j + 1];
							sev[findRoot(P, imgLabels_prev[j + 1])] = 1;
							goto tree_B;
						}
						else {
							imgLabels_row[j] = lunique; P[lunique] = lunique; sev[lunique] = 1; lunique++;
							goto tree_B;
						}
					}
				}
				else {
					//weak edge
					if (condition_q) {
						imgLabels_row[j] = imgLabels_prev[j];
						goto tree_B;
					}
					else {
						if (condition_r) {
							imgLabels_row[j] = imgLabels_prev[j + 1];
							goto tree_B;
						}
						else {
							imgLabels_row[j] = lunique; P[lunique] = lunique; sev[lunique] = 0; lunique++;
							goto tree_B;
						}
					}
				}
			}
			else {
				//background
				imgLabels_row[j] = 0;
				goto tree_A;
			}
		}
		else {
			//background
			imgLabels_row[j] = 0;
			goto tree_A;
		}

	tree_A: if (++j >= w - 1) goto break_A;
		mag = mag_tbl[++index2];
		if (mag > th_low) {
			dx = dx_tbl[index2];
			dy = dy_tbl[index2];
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

			bMaxima = true;
			//non-maxima suppression
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
			if (bMaxima) {
				if (mag > th_high) {
					//strong edge
					if (condition_q) {
						imgLabels_row[j] = imgLabels_prev[j];
						sev[findRoot(P, imgLabels_prev[j])] = 1;
						goto tree_B;
					}
					else {
						if (condition_p) {
							if (condition_r) {
								imgLabels_row[j] = set_union_Strong(P, sev, imgLabels_prev[j - 1], imgLabels_prev[j + 1]);
								goto tree_B;
							}
							else {
								sev[findRoot(P, imgLabels_prev[j - 1])] = 1;
								imgLabels_row[j] = imgLabels_prev[j - 1];
								goto tree_B;
							}
						}
						else {
							if (condition_r) {
								sev[findRoot(P, imgLabels_prev[j + 1])] = 1;
								imgLabels_row[j] = imgLabels_prev[j + 1];
								goto tree_B;
							}
							else {
								imgLabels_row[j] = lunique; P[lunique] = lunique; sev[lunique] = 1; lunique++;
								goto tree_B;
							}
						}
					}
				}
				else {
					//weak edge
					if (condition_q) {
						imgLabels_row[j] = imgLabels_prev[j];
						goto tree_B;
					}
					else {
						if (condition_p) {
							if (condition_r) {
								imgLabels_row[j] = set_union_Weak(P, sev, imgLabels_prev[j - 1], imgLabels_prev[j + 1]);
								goto tree_B;
							}
							else {
								imgLabels_row[j] = imgLabels_prev[j - 1];
								goto tree_B;
							}
						}
						else {
							if (condition_r) {
								imgLabels_row[j] = imgLabels_prev[j + 1];
								goto tree_B;
							}
							else {
								imgLabels_row[j] = lunique; P[lunique] = lunique; sev[lunique] = 0; lunique++;
								goto tree_B;
							}
						}
					}
				}
			}
			else {
				//background
				imgLabels_row[j] = 0;
				goto tree_A;
			}
		}
		else {
			//background
			imgLabels_row[j] = 0;
			goto tree_A;
		}

	tree_B: if (++j >= w - 1) goto break_B;
		mag = mag_tbl[++index2];
		if (mag > th_low) {
			dx = dx_tbl[index2];
			dy = dy_tbl[index2];
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

			bMaxima = true;
			//non-maxima suppression
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
			if (bMaxima) {
				if (mag > th_high) {
					//strong edge
					if (condition_q) {
						sev[findRoot(P, imgLabels_prev[j])] = 1;
						imgLabels_row[j] = imgLabels_prev[j];
						goto tree_B;
					}
					else {
						if (condition_r) {
							imgLabels_row[j] = set_union_Strong(P, sev, imgLabels_prev[j + 1], imgLabels_row[j - 1]);
							goto tree_B;
						}
						else {
							sev[findRoot(P, imgLabels_row[j - 1])] = 1;
							imgLabels_row[j] = imgLabels_row[j - 1];
							goto tree_B;
						}
					}
				}
				else {
					//weak edge
					if (condition_q) {
						imgLabels_row[j] = imgLabels_prev[j];
						goto tree_B;
					}
					else {
						if (condition_r) {
							imgLabels_row[j] = set_union_Weak(P, sev, imgLabels_prev[j + 1], imgLabels_row[j - 1]);
							goto tree_B;
						}
						else {
							imgLabels_row[j] = imgLabels_row[j - 1];
							goto tree_B;
						}
					}
				}
			}
			else {
				//background
				imgLabels_row[j] = 0;
				goto tree_A;
			}
		}
		else {
			//background
			imgLabels_row[j] = 0;
			goto tree_A;
		}
	break_A:
		imgLabels_row[j] = 0;
		continue;
	break_B:
		imgLabels_row[j] = 0;
		continue;
	}

	uint nLabel = flattenL_Canny(P, sev, lunique);

	//second scan
	for (int r_i = 0; r_i < imgLabels.rows; ++r_i) {
		uint * b = imgLabels.ptr<uint>(r_i);
		uint * const e = b + imgLabels.cols;
		for (; b != e; ++b) {
			*b = P[*b];
		}
	}

	fastFree(P);
	fastFree(sev);

	fastFree(mag_tbl);
	fastFree(dx_tbl);
	fastFree(dy_tbl);
	return nLabel;

	return 0;
}