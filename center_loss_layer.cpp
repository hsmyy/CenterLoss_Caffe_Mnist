#include <vector>

#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template<typename Dtype>
	void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());

		int channels = bottom[0]->channels();
		int num = bottom[0]->num();

		alpha = this->layer_param_.center_loss_param().alpha();
		lossWeight = this->layer_param_.center_loss_param().loss_weight();
		clusterNum = this->layer_param_.center_loss_param().cluster_num();

		center_info_.Reshape(clusterNum, channels, 1, 1);
		center_loss_.Reshape(num, channels, 1, 1);
		center_update_count_.resize(clusterNum);
        caffe_set(clusterNum * channels, Dtype(0.0), center_info_.mutable_cpu_data());
	}

	template<typename Dtype>
	void CenterLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*> &bottom,
		const vector<Blob<Dtype>*> &top){
		const Dtype *feature = bottom[0]->cpu_data();
		const Dtype *label = bottom[1]->cpu_data();
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		Dtype loss = 0;
		caffe_set(clusterNum * channels, Dtype(0.0), center_info_.mutable_cpu_diff());
		for(int i = 0; i < clusterNum; ++i){
			center_update_count_[i] = 1;
		}
		for(int i = 0; i < num; ++i){
			int targetLabel = label[i];
			caffe_sub(channels, feature + i * channels,
				center_info_.cpu_data() + targetLabel * channels,
				center_loss_.mutable_cpu_data() + i * channels);
			// store the update loss and number
			caffe_add(channels, center_loss_.cpu_data() + i * channels,
				center_info_.cpu_diff() + targetLabel * channels,
                center_info_.mutable_cpu_diff() + targetLabel * channels);
			center_update_count_[targetLabel]++;

			loss += caffe_cpu_dot(channels, center_loss_.cpu_data() + i * channels,
				center_loss_.cpu_data() + i * channels) * lossWeight / Dtype(2.0) / static_cast<Dtype>(num);
		}
		top[0]->mutable_cpu_data()[0] = loss;
		// update center loss.
		for(int i = 0; i < clusterNum; ++i){
			Dtype scale = -alpha * lossWeight / Dtype(center_update_count_[i]);
			caffe_scal(channels, scale, center_info_.mutable_cpu_diff() + i * channels);
		}
		center_info_.Update();
	}

	template<typename Dtype>
	void CenterLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*> &top,
		const vector<bool> &propagate_down,
		const vector<Blob<Dtype>*> &bottom){
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		caffe_scal(num * channels, lossWeight, center_loss_.mutable_cpu_data());
        Dtype *out = bottom[0]->mutable_cpu_diff();
		caffe_copy(num * channels, center_loss_.cpu_data(), out);

	}
	template <typename Dtype>
	void CenterLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Forward_cpu(bottom, top);
	}
	template<typename Dtype>
	void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		Backward_cpu(top, propagate_down,bottom);
	}
	INSTANTIATE_CLASS(CenterLossLayer);
	REGISTER_LAYER_CLASS(CenterLoss);
}
