#include "IBI_S2DC.h"

/**************RANSAC************/
//Hypothesis generation
void Group::RANSAC_trans_est(pcl::PointXYZ& point_s1, pcl::PointXYZ& point_s2, pcl::PointXYZ& point_s3, pcl::PointXYZ& point_t1, pcl::PointXYZ& point_t2, pcl::PointXYZ& point_t3, Eigen::Matrix4f& Mat)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_target(new pcl::PointCloud<pcl::PointXYZ>);

	LRF_source->points.push_back(point_s1); LRF_source->points.push_back(point_s2); LRF_source->points.push_back(point_s3);//LRF_source->points.push_back(s_4);
	LRF_target->points.push_back(point_t1); LRF_target->points.push_back(point_t2); LRF_target->points.push_back(point_t3);//LRF_source->points.push_back(t_4);
	//
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> SVD;
	SVD.estimateRigidTransformation(*LRF_source, *LRF_target, Mat);
}
//Hypothesis quality estimation
double Group::RANSAC_mae(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float inlier_threshold)
{
	double mae = 0.0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_match_points, *source_match_points_trans, Mat);
	for (int i = 0; i < source_match_points_trans->points.size(); i++)
	{
		double X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
		double Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
		double Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
		double dist = sqrt(X * X + Y * Y + Z * Z);
		double mae_temp;
		if (dist < inlier_threshold)
		{
			mae_temp = (inlier_threshold - dist) / inlier_threshold ;
			mae += mae_temp;
		}
	}
	return mae;
}
//RANSAC
int Group::RANSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Corres>& Match, float RANSAC_inlier_judge_thresh, int _Iterations, Eigen::Matrix4f& Mat)
{
	Mat = Eigen::Matrix4f::Identity();
	double mae = 0.0;
	int x = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = cloud_source->points[Match[i].source_idx];
		point_t = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	int Iterations = _Iterations;
	int Rand_seed = Iterations;
	for (int i = 0; i < Iterations; i++)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		point_s1 = cloud_source->points[Match[Match_Idx1].source_idx];
		point_s2 = cloud_source->points[Match[Match_Idx2].source_idx];
		point_s3 = cloud_source->points[Match[Match_Idx3].source_idx];
		point_t1 = cloud_target->points[Match[Match_Idx1].target_idx];
		point_t2 = cloud_target->points[Match[Match_Idx2].target_idx];
		point_t3 = cloud_target->points[Match[Match_Idx3].target_idx];
		//
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		double mae_iter = RANSAC_mae(source_match_points, target_match_points, Mat_iter, RANSAC_inlier_judge_thresh);
		if (mae_iter > mae)
		{
			mae = mae_iter;
			Mat = Mat_iter;
			x = i;
		}
	}
	return 1;
}