#include "IBI_S2DC.h"

/**************3D Hough voting************/
void Group::HoughVoting_Corres_group(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Corres>& Match, vector<Corres> &Match_filteled, int Hough_bin_num)
{
	int i, j;
	//model centroid
	Eigen::Vector3f centroid(0, 0, 0);
	for (i = 0; i < cloud_source->points.size(); i++)
	{
		centroid += cloud_source->at(i).getVector3fMap();
	}
	centroid /= static_cast<float> (cloud_source->points.size());
	//model votes
	vector<Eigen::Vector3f> model_votes;
	model_votes.resize(Match.size());
	for (i = 0; i < Match.size(); i++)
	{
		Eigen::Vector3f x_ax(Match[i].source_LRF.x_axis.x, Match[i].source_LRF.x_axis.y, Match[i].source_LRF.x_axis.z);
		Eigen::Vector3f y_ax(Match[i].source_LRF.y_axis.x, Match[i].source_LRF.y_axis.y, Match[i].source_LRF.y_axis.z);
		Eigen::Vector3f z_ax(Match[i].source_LRF.z_axis.x, Match[i].source_LRF.z_axis.y, Match[i].source_LRF.z_axis.z);

		model_votes[i].x() = x_ax.dot(centroid - cloud_source->at(Match[i].source_idx).getVector3fMap());
		model_votes[i].y() = y_ax.dot(centroid - cloud_source->at(Match[i].source_idx).getVector3fMap());
		model_votes[i].z() = z_ax.dot(centroid - cloud_source->at(Match[i].source_idx).getVector3fMap());
	}
	//scene votes
	vector<Vertex> hough_votes;
	hough_votes.resize(Match.size());
	float x_min, x_max, y_min, y_max, z_min, z_max, x_step, y_step, z_step;
	vector<float> x_values, y_values, z_values;
	for (i = 0; i < Match.size(); i++)
	{
		const Eigen::Vector3f& scene_point = cloud_target->at(Match[i].target_idx).getVector3fMap();
		Eigen::Vector3f scene_point_lrf_x(Match[i].target_LRF.x_axis.x, Match[i].target_LRF.x_axis.y, Match[i].target_LRF.x_axis.z);
		Eigen::Vector3f scene_point_lrf_y(Match[i].target_LRF.y_axis.x, Match[i].target_LRF.y_axis.y, Match[i].target_LRF.y_axis.z);
		Eigen::Vector3f scene_point_lrf_z(Match[i].target_LRF.z_axis.x, Match[i].target_LRF.z_axis.y, Match[i].target_LRF.z_axis.z);
		hough_votes[i].x = scene_point_lrf_x[0] * model_votes[i].x() + scene_point_lrf_y[0] * model_votes[i].y() + scene_point_lrf_z[0] * model_votes[i].z() + scene_point.x();
		hough_votes[i].y = scene_point_lrf_x[1] * model_votes[i].x() + scene_point_lrf_y[1] * model_votes[i].y() + scene_point_lrf_z[1] * model_votes[i].z() + scene_point.y();
		hough_votes[i].z = scene_point_lrf_x[2] * model_votes[i].x() + scene_point_lrf_y[2] * model_votes[i].y() + scene_point_lrf_z[2] * model_votes[i].z() + scene_point.z();
		x_values.push_back(hough_votes[i].x);
		y_values.push_back(hough_votes[i].y);
		z_values.push_back(hough_votes[i].z);
	}

	/*FILE*fp=fopen("D:\\Exp_3Dcorres\\Hough_cloud.xyz","w");
	fprintf(fp,"%d\n",hough_votes.size());
	for(i=0;i<hough_votes.size();i++)
	{
		fprintf(fp,"%f %f %f\n",hough_votes[i].x,hough_votes[i].y,hough_votes[i].z);
	}
	fclose(fp);*/

	sort(x_values.begin(), x_values.end());
	sort(y_values.begin(), y_values.end());
	sort(z_values.begin(), z_values.end());
	x_min = x_values[0]; x_max = x_values[x_values.size() - 1];
	y_min = y_values[0]; y_max = y_values[y_values.size() - 1];
	z_min = z_values[0]; z_max = z_values[z_values.size() - 1];
	x_step = (x_max - x_min) / Hough_bin_num;
	y_step = (y_max - y_min) / Hough_bin_num;
	z_step = (z_max - z_min) / Hough_bin_num;
	//Hough space
	vector<vector<int>> sub_space_Corre_IDs;
	sub_space_Corre_IDs.resize(Hough_bin_num*Hough_bin_num*Hough_bin_num);//cubic space
	for (i = 0; i < Match.size(); i++)
	{
		Vertex vote_temp = hough_votes[i];
		int Idx_x = (vote_temp.x - x_min) / x_step;
		int Idx_y = (vote_temp.y - y_min) / y_step;
		int Idx_z = (vote_temp.z - z_min) / z_step;
		if (Idx_x < 0) Idx_x = 0; if (Idx_x > (Hough_bin_num - 1)) Idx_x = Hough_bin_num - 1;
		if (Idx_y < 0) Idx_y = 0; if (Idx_y > (Hough_bin_num - 1)) Idx_y = Hough_bin_num - 1;
		if (Idx_z < 0) Idx_z = 0; if (Idx_z > (Hough_bin_num - 1)) Idx_z = Hough_bin_num - 1;
		int Idx = Idx_z * Hough_bin_num*Hough_bin_num + Idx_y * Hough_bin_num + Idx_x;
		sub_space_Corre_IDs[Idx].push_back(i);
	}
	vector<int> sub_space_sizes;
	for (i = 0; i < sub_space_Corre_IDs.size(); i++)
		sub_space_sizes.push_back(sub_space_Corre_IDs[i].size());
	sort(sub_space_sizes.begin(), sub_space_sizes.end());
	int max_size = sub_space_sizes[sub_space_sizes.size() - 1];
	vector<int> Peek_IDs;//1peek+6neighbors
	Peek_IDs.resize(7);
	for (i = 0; i < sub_space_Corre_IDs.size(); i++)
	{
		if (sub_space_Corre_IDs[i].size() == max_size)
		{
			Peek_IDs[0] = i;
			Peek_IDs[1] = i + 1;
			Peek_IDs[2] = i - 1;
			Peek_IDs[3] = i + Hough_bin_num;
			Peek_IDs[4] = i - Hough_bin_num;
			Peek_IDs[5] = i + Hough_bin_num * Hough_bin_num;
			Peek_IDs[6] = i - Hough_bin_num * Hough_bin_num;
		}
	}
	for (i = 0; i < Peek_IDs.size(); i++)
	{
		if ((Peek_IDs[i] >= 0) && (Peek_IDs[i] < sub_space_Corre_IDs.size()))
		{
			for (j = 0; j < sub_space_Corre_IDs[Peek_IDs[i]].size(); j++)
				Match_filteled.push_back(Match[sub_space_Corre_IDs[Peek_IDs[i]][j]]);
		}
	}
}