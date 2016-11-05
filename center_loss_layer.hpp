#ifndef CAFFE_CENTER_LOSS_LAYER_HPP_
#define CAFFE_CENTER_LOSS_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe{
	/*
	NOTE:currently we won't store center info.
	*/
template<typename Dtype>
class CenterLossLayer: public LossLayer<Dtype>{
public:
	explicit CenterLossLayer(const LayerParameter &param): LossLayer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual inline int ExactNumBottomBlobs() const {return 2;}
	virtual inline const char *type() const {return "CenterLoss";}
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return bottom_index != 1;
	}
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> center_loss_;
	Blob<Dtype> center_info_;
	vector<int> center_update_count_;
	Dtype alpha;
	Dtype lossWeight;
	int clusterNum;
};

}

#endif
