#include "IBI_S2DC.h"

//Synthetic corres by outlier ratio
pcl::PointCloud<pcl::PointXYZ> Syn_corr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, vector<Eigen::Matrix4f>& gts, float min, float max, vector<Corres>& Match_up)
{
	int K = gts.size();
	pcl::PointCloud<pcl::PointXYZ> cloud_src, cloud_tar, cloud_trs, cloud_nos;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_noise(new pcl::PointCloud<pcl::PointXYZ>);
	cloud_noise->points.resize(1000);
	for (size_t i = 0; i < cloud_noise->points.size(); ++i)
	{
		cloud_noise->points[i].x = 10 * rand() / (RAND_MAX + 1.0f);
		cloud_noise->points[i].y = 10 * rand() / (RAND_MAX + 1.0f);
		cloud_noise->points[i].z = 10 * rand() / (RAND_MAX + 1.0f);
	}
	cloud_nos = *cloud_noise;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target;
	cloud_src = *cloud_source;
	int src_num = cloud_src.size();
	for (int t = 0; t < gts.size(); t++)
	{
		pcl::transformPointCloud(cloud_src, cloud_trs, gts[t]);
		for (int c = 0; c < cloud_trs.size(); c++)
		{
			cloud_trs[c].x += cloud_trs[c].x*static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.1) - 0.05;
			cloud_trs[c].y += cloud_trs[c].y*static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.1) - 0.05;
			cloud_trs[c].z += cloud_trs[c].z*static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.1) - 0.05;
			Corres corres;
			corres.source_idx = c;
			corres.target_idx = c + t * src_num;
			Match_up.push_back(corres);
		}
		cloud_tar += cloud_trs;
	}
	float outlier_ratio = min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
	float inlier_ratio = 1 - outlier_ratio;
	int inlier_num = cloud_tar.size();
	int outlier_num = inlier_num * (outlier_ratio / inlier_ratio);
	cloud_tar += cloud_nos;
	int tar_num = cloud_tar.size();
	//cout << "tar_num = " << tar_num << endl;
	int i = 0;
	while (i < outlier_num)
	{
		Corres corres;
		int src_idx = static_cast <int> (rand()) / (static_cast <int> (RAND_MAX / src_num));
		corres.source_idx = src_idx;
		int tar_idx = static_cast <int> (rand()) / (static_cast <int> (RAND_MAX / tar_num));
		corres.target_idx = tar_idx;
		if ((src_idx <= src_num) && (tar_idx <= tar_num))
		{
			Match_up.push_back(corres);
			i++;
		}
	}
	return cloud_tar;
}