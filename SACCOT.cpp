#include "IBI_S2DC.h"

/**************SACCOT************/
//Get nodes by match
void Group::getNodeByMatch(vector<Corres>& Match)
{
	for (int i = 0; i < Match.size(); i++)
	{
		Node node;
		node.source_idx = Match[i].source_idx;
		node.target_idx = Match[i].target_idx;
		node.degree = 0;
		node.index = i;
		nodes.push_back(node);
	}
}
//Compute distance between two nodes
double computeDistance(pcl::PointXYZ &point_i, pcl::PointXYZ &point_j)
{
	float distance;
	distance = sqrt(((point_i.x - point_j.x)*(point_i.x - point_j.x) + (point_i.y - point_j.y)*(point_i.y - point_j.y) + (point_i.z - point_j.z)*(point_i.z - point_j.z)));
	return distance;
}
//Compute rigidity r(ci,cj)
float computeRigidity(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j)
{
	float rigidity;
	rigidity = abs(computeDistance(source_i, source_j) - computeDistance(target_i, target_j));
	return rigidity;
}
//Compute compatibility
double computeCompatibility(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j)
{
	double r, R, T, n1, N1, T1, x, y, z;
	double compatibility;
	r = computeRigidity(source_i, source_j, target_i, target_j);
	R = r * r;
	T = 2 * 10 * 10;//tcons=10
	compatibility = exp(-(R / T));
	return compatibility;
}

//Compute relationship information of graph
void Group::computeMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar, vector<Corres>& Match, double compatibility_threshold)
{
	double compatibility;
	int source_match_idxi, target_match_idxi, source_match_idxj, target_match_idxj;
	pcl::Normal normal_srci, normal_tari, normal_srcj, normal_tarj;
	pcl::PointXYZ source_i, source_j, target_i, target_j;
	compatibility_matrix = vector<vector<double>>(nodes.size(), vector<double>(nodes.size(), 0));
	for (int i = 0; i < Match.size(); i++)//Select the first match
	{
		source_match_idxi = Match[i].source_idx;
		target_match_idxi = Match[i].target_idx;//Get the index of source and target pointcloud
		source_i = cloud_src->points[source_match_idxi];
		target_i = cloud_tar->points[target_match_idxi];
		for (int j = 0; j < Match.size(); j++)//Select the second match
		{
			if (i != j)//Continue while two matches are different
			{
				source_match_idxj = Match[j].source_idx;
				target_match_idxj = Match[j].target_idx;//Get the index of source and target pointcloud
				source_j = cloud_src->points[source_match_idxj];
				target_j = cloud_tar->points[target_match_idxj];
				//Compute compatibility distance
				compatibility = computeCompatibility(source_i, source_j, target_i, target_j);
				if (compatibility >= compatibility_threshold)//means two matches are compatible
				{
					nodes[i].degree++;//The degree of each node
					compatibility_matrix[i][j] = compatibility;//Fill the matrix with each compatibility distance
				}
			}
		}
	}
}

//Compute Degree and Compatibility score of cots
vector<COT> Group::computeCOT(int topN)
{
	set<vector<int>> COTset;
	for (int i = 0; i < nodes.size(); i++)
	{
		vector<int>connect_nodes;
		for (int j = 0; j < nodes.size(); j++)
		{
			if (abs(compatibility_matrix[i][j] - 0) > 1e-15)connect_nodes.push_back(j);//Put all nodes connected with i in connect_nodes
			if (i == j)continue;
		}
		for (int j : connect_nodes)//Determine the 1st line of a cot
		{
			for (int k : connect_nodes)//Determine the 2nd line of a cot
			{
				if (k == i || k == j)continue;
				if (abs(compatibility_matrix[j][k] - 0) > 1e-15)//Determine the 3rd line of a cot
				{
					vector<int> label;
					label.push_back(i);
					label.push_back(j);
					label.push_back(k);
					sort(label.begin(), label.end());//Rank three index in ascending order of label
					COTset.insert(label);//Insert all the labels of cot into COTset
				}
			}
		}
	}
	for (set<vector<int>>::iterator it = COTset.begin(); it != COTset.end(); it++)
	{
		COT cot;
		cot.index1 = (*it)[0];
		cot.index2 = (*it)[1];
		cot.index3 = (*it)[2];
		cot.degree = nodes[cot.index1].degree + nodes[cot.index2].degree + nodes[cot.index3].degree;//Deg(COT)
		cot.compatibility = compatibility_matrix[cot.index1][cot.index2] + compatibility_matrix[cot.index1][cot.index3] + compatibility_matrix[cot.index2][cot.index3];//Comp(COT)
		cots.push_back(cot);
	}
	sort(cots.begin(), cots.end(), [](COT cot1, COT cot2) {return cot1.compatibility > cot2.compatibility; });//Rank cots in descending order of Compatibility
//    sort(cots.begin(), cots.end(), [](COT cot1, COT cot2) {return cot1.degree > cot2.degree; });//Rank cots in descending order of Degree
	cots.resize(topN);//N=500
	for (int m = 0; m < cots.size(); m++)
	{
		Corres pair1;
		Corres pair2;
		Corres pair3;
		pair1.source_idx = nodes[cots[m].index1].source_idx;
		pair1.target_idx = nodes[cots[m].index1].target_idx;
		pair2.source_idx = nodes[cots[m].index2].source_idx;
		pair2.target_idx = nodes[cots[m].index2].target_idx;
		pair3.source_idx = nodes[cots[m].index3].source_idx;
		pair3.target_idx = nodes[cots[m].index3].target_idx;
		match1.push_back(pair1);
		match2.push_back(pair2);
		match3.push_back(pair3);
	}
	return cots;
}
//Get top N triplets
vector<Triplet> Group::computeTriplet(int topN)
{
	for (int i = 0; i < nodes.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			for (int k = 0; k < j; k++)
			{
				Triplet triplet;
				triplet.index1 = nodes[i].index;
				triplet.index2 = nodes[j].index;
				triplet.index3 = nodes[k].index;
				triplet.degree = nodes[i].degree + nodes[j].degree + nodes[k].degree;
				triplets.push_back(triplet);
			}
		}
	}
	sort(triplets.begin(), triplets.end(), [](Triplet triplet1, Triplet triplet2) {return triplet1.degree > triplet2.degree; });//Rank all the triplets in descending order of Degree
	triplets.resize(topN);
	return triplets;
}
int Group::SAC_COT(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Corres>& Match, float RANSAC_inlier_judge_thresh, int _Iterations, Eigen::Matrix4f& Mat)
{
	double mae = 0.0;
	Mat = Eigen::Matrix4f::Identity();
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = cloud_src->points[Match[i].source_idx];
		point_t = cloud_tar->points[Match[i].target_idx];
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
		int COT_Idx1, COT_Idx2, COT_Idx3;
		if (i <= cots.size())
		{//Use all m cots for iteration
			COT_Idx1 = cots[i].index1;
			COT_Idx2 = cots[i].index2;
			COT_Idx3 = cots[i].index3;
		}
		else
		{//Replace the insufficient cots by top(N - m) triplets
			COT_Idx1 = triplets[i - cots.size()].index1;
			COT_Idx2 = triplets[i - cots.size()].index2;
			COT_Idx3 = triplets[i - cots.size()].index3;
		}
		point_s1 = cloud_src->points[Match[COT_Idx1].source_idx];
		point_s2 = cloud_src->points[Match[COT_Idx2].source_idx];
		point_s3 = cloud_src->points[Match[COT_Idx3].source_idx];
		point_t1 = cloud_tar->points[Match[COT_Idx1].target_idx];
		point_t2 = cloud_tar->points[Match[COT_Idx2].target_idx];
		point_t3 = cloud_tar->points[Match[COT_Idx3].target_idx];

		Eigen::Matrix4f Mat_iter;
		Mat_iter = Eigen::Matrix4f::Identity();
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		double mae_iter = RANSAC_mae(source_match_points, target_match_points, Mat_iter, RANSAC_inlier_judge_thresh);
		if (mae_iter > mae)
		{
			mae = mae_iter;
			Mat = Mat_iter;
		}
	}
	return 1;
}