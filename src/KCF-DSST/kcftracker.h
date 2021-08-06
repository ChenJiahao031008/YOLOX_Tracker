/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: horizontal area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#pragma once

#include "tracker.h"

#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif

class KCFTracker : public Tracker
{
public:
    // Constructor
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    // 第二种初始化方式
    KCFTracker(){};
    void SetKCFParam(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    virtual ~KCFTracker();

    // Initialize tracker
    virtual void init(const cv::Rect &roi, cv::Mat image);

    virtual void init(const cv::Point pt1, const cv:: Point pt2, cv::Mat image);

    // Update position based on the new frame
    virtual cv::Rect update(cv::Mat image);

    float interp_factor; // linear interpolation factor for adaptation
    float sigma; // gaussian kernel bandwidth
    float lambda; // regularization
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    float padding; // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size

    int base_width; // initial ROI widt
    int base_height; // initial ROI height
    int scale_max_area; // max ROI size before compressing
    float scale_padding; // extra area surrounding the target for scaling
    float scale_step; // scale step for multi-scale estimation
    float scale_sigma_factor; // bandwidth of gaussian target
    int n_scales; // # of scaling windows
    float scale_lr; // scale learning rate
    float *scaleFactors; // all scale changing rate, from larger to smaller with 1 to be the middle
    int scale_model_width; // the model width for scaling
    int scale_model_height; // the model height for scaling
    float currentScaleFactor; // scaling rate
    float min_scale_factor; // min scaling rate
    float max_scale_factor; // max scaling rate
    float scale_lambda; // regularization

    int successflag = 0;
    int tmpflag = 0;

  protected:
    // Detect object in the current frame.
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);

    // train tracker with a single image
    void train(cv::Mat x, float train_interp_factor);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

    // Initialize Hanning window. Function called only in the first frame.
    void createHanningMats();

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

    // Compute the FFT Guassian Peak for scaling
    cv::Mat computeYsf();

    // Compute the hanning window for scaling
    cv::Mat createHanningMatsForScale();

    // Initialization for scales
    int dsstInit(const cv::Rect &roi, cv::Mat image);

    // Compute the F^l in the paper
    cv::Mat get_scale_sample(const cv::Mat & image);

    // Update the ROI size after training
    void update_roi();

    // Train method for scaling
    int train_scale(cv::Mat image, bool ini = false);

    // Detect the new scaling rate
    cv::Point2i detect_scale(cv::Mat image);

    cv::Mat _alphaf;
    cv::Mat _prob;
    cv::Mat _tmpl;
    cv::Mat _num;
    cv::Mat _den;
    cv::Mat _labCentroids;

    cv::Mat sf_den;
    cv::Mat sf_num;

private:
    int size_patch[3];
    cv::Mat hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;

    cv::Mat s_hann;
    cv::Mat ysf;

};
