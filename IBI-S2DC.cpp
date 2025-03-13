#include "IBI_S2DC.h"

//Compute PointCloud resolution
double computeResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	double resolution = 0.0;
	int n_points = 0;
	int nresolution;
	std::vector<int> indices;
	std::vector<float> sqr_distances;
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!pcl_isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nresolution = tree.nearestKSearch(cloud->points[i], 2, indices, sqr_distances);//return :number of neighbors found
		if (nresolution == 2)
		{
			resolution += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		resolution /= n_points;
	}
	return resolution;
}

//Generate Label for corres
vector<bool> Label0(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Corres>& Match, float label_thresh, vector<Eigen::Matrix4f>& gts)
{
	vector<bool> labels;
	for (int t = 0; t < gts.size(); t++)
	{
		vector<bool> Label;
		pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Src(new pcl::PointCloud <pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Tar(new pcl::PointCloud <pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Trans(new pcl::PointCloud <pcl::PointXYZ>);
		pcl::PointXYZ Point_Temp_s, Point_Temp_t;
		for (int i = 0; i < Match.size(); i++)
		{
			Point_Temp_s = cloud_src->points[Match[i].source_idx];
			Point_Temp_t = cloud_tar->points[Match[i].target_idx];
			keyPointCloud_Src->push_back(Point_Temp_s);
			keyPointCloud_Tar->push_back(Point_Temp_t);
		}
		pcl::transformPointCloud(*keyPointCloud_Src, *keyPointCloud_Trans, gts[t]);
		for (int i = 0; i < Match.size(); i++)
		{
			float x1 = keyPointCloud_Trans->points[i].x;
			float y1 = keyPointCloud_Trans->points[i].y;
			float z1 = keyPointCloud_Trans->points[i].z;
			float x2 = keyPointCloud_Tar->points[i].x;
			float y2 = keyPointCloud_Tar->points[i].y;
			float z2 = keyPointCloud_Tar->points[i].z;
			float dis = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
			if (dis <= label_thresh) Match[i].inlier = true;
		}
		for (int i = 0; i < Match.size(); i++) labels.push_back(Match[i].inlier);
	}
	return labels;
}
int Group::Inlier(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Corres>& Match, float inlier_thresh, Eigen::Matrix4f& gt)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Src(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Tar(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Trans(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointXYZ Point_Temp_s, Point_Temp_t;
	for (int i = 0; i < Match.size(); i++)
	{
		Point_Temp_s = cloud_src->points[Match[i].source_idx];
		Point_Temp_t = cloud_tar->points[Match[i].target_idx];
		keyPointCloud_Src->push_back(Point_Temp_s);
		keyPointCloud_Tar->push_back(Point_Temp_t);
	}
	pcl::transformPointCloud(*keyPointCloud_Src, *keyPointCloud_Trans, gt);
	int num = 0;
	for (int i = 0; i < Match.size(); i++)
	{
		float x1 = keyPointCloud_Trans->points[i].x;
		float y1 = keyPointCloud_Trans->points[i].y;
		float z1 = keyPointCloud_Trans->points[i].z;
		float x2 = keyPointCloud_Tar->points[i].x;
		float y2 = keyPointCloud_Tar->points[i].y;
		float z2 = keyPointCloud_Tar->points[i].z;
		float dis = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
		if (dis <= inlier_thresh)num++;
	}
	return num;
}

/**************DownSampling************/
//Random number generation
void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx)
{
	boost::mt19937 engine(seed);
	boost::uniform_int<> distribution(start, end);
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > myrandom(engine, distribution);

	for (int i = 0; i < rand_num; i++)
	{
		int new_num = myrandom();
		while (find(idx.begin(), idx.end(), new_num) != idx.end())
		{
			new_num = myrandom();
		}
		idx.push_back(new_num);
	}
}
void Rand_x(int seed, int ds, vector<int>& output)
{
	int start = 0;
	int end = ds - 1;
	boost_rand(seed, start, end, ds, output);
}
void Rand_3(int seed, int scale, int& output1, int& output2, int& output3)
{
	std::vector<int> result;
	int start = 0;
	int end = scale - 1;
	boost_rand(seed, start, end, scale, result);
	output1 = result[0];
	output2 = result[1];
	output3 = result[2];
}
//downsample 1024 corres
void downSampling(vector<Corres>& Match_up, vector<Corres>& Match, int downSize, vector<int>& target_idxs)
{
	vector<int> indices;
	Rand_x(123456, downSize, indices);
	for (int idx : indices) 
	{
		Match.push_back(Match_up[idx]);
		target_idxs.push_back(Match_up[idx].target_idx);
	} 
}

/**************GTM************/
//Compute payoff matrix M(i,j)
void Group::computepayoffMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar, vector<Corres>& Match, Eigen::MatrixXf& M)
{
	int i, j;
	//Eigen::MatrixXf M(Match.size(), Match.size());
	for (i = 0; i < Match.size(); i++)
	{
		for (j = 0; j < Match.size(); j++)
		{
			if (i == j)
				M(i, j) = 0.0f;
			if (i < j)
			{
				float x1 = cloud_src->points[Match[i].source_idx].x - cloud_src->points[Match[j].source_idx].x;
				float y1 = cloud_src->points[Match[i].source_idx].y - cloud_src->points[Match[j].source_idx].y;
				float z1 = cloud_src->points[Match[i].source_idx].z - cloud_src->points[Match[j].source_idx].z;
				float x2 = cloud_tar->points[Match[i].target_idx].x - cloud_tar->points[Match[j].target_idx].x;
				float y2 = cloud_tar->points[Match[i].target_idx].y - cloud_tar->points[Match[j].target_idx].y;
				float z2 = cloud_tar->points[Match[i].target_idx].z - cloud_tar->points[Match[j].target_idx].z;
				float a = sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2));
				float b = sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2));
				if ((a != 0.0) && (b != 0.0))
				{
					M(i, j) = a / b;
					if (M(i, j) > b / a) M(i, j) = b / a;
				}
				else
					M(i, j) = 0;
			}
			if (i > j) M(i, j) = M(j, i);
		}
	}
}
//Select a highly compatible set of Corres
void Group::GTM_Corres_select(int Iterations, vector<Corres>& Match, vector<Corres>& Match_up, Eigen::MatrixXf& M, vector<Corres>& Match_inlier)
{
	int i, j;
	vector<int> target_idxs;
	Eigen::VectorXf V(Match.size());//population
	for (i = 0; i < Match.size(); i++)
	{
		V(i) = 1.0 / Match.size();
	}
	//    cout << V << endl;
	for (i = 0; i < Iterations; i++)
	{
		Eigen::VectorXf UP(Match.size());
		UP = M * V;
		//cout << UP << endl;
		float down = V.transpose()*UP;
		for (j = 0; j < Match.size(); j++)
		{
			float up = UP(j);
			V(j) = V(j)*up / down;
		}
	}
	vector<float> values;
	for (i = 0; i < Match.size(); i++) values.push_back(V(i));
	float thresh = OSTU_thresh_GTM(values);
	for (i = 0; i < Match.size(); i++)
	{
		if (V[i] > thresh)
		{
			Match[i].score=V[i];
			Match_inlier.push_back(Match[i]);
		}
	}
}

/**************RV************/
//Compute pariwise consistency
float Group::computePariwiseConsistency(PointCloudPtr cloud_source, PointCloudPtr cloud_target, float mr, float RANSAC_inlier_judge_thresh, Corres& c, vector<Corres>& Voting_set)
{
	int i, j;
	float consistency_score = 0.0f;
	//
	for (i = 0; i < Voting_set.size(); i++)
	{
		float d1, d2, d3, d4;
		float consistency_score_temp = 0.0f;
		float consistency_score_temp2 = 0.0f;
		//d1
		float x1 = cloud_source->points[c.source_idx].x - cloud_source->points[Voting_set[i].source_idx].x;
		float y1 = cloud_source->points[c.source_idx].y - cloud_source->points[Voting_set[i].source_idx].y;
		float z1 = cloud_source->points[c.source_idx].z - cloud_source->points[Voting_set[i].source_idx].z;
		float x2 = cloud_target->points[c.target_idx].x - cloud_target->points[Voting_set[i].target_idx].x;
		float y2 = cloud_target->points[c.target_idx].y - cloud_target->points[Voting_set[i].target_idx].y;
		float z2 = cloud_target->points[c.target_idx].z - cloud_target->points[Voting_set[i].target_idx].z;
		float dist1 = sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2));
		float dist2 = sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2));
		d1 = abs(dist1 - dist2) / (10.0*mr);
	
		consistency_score_temp = exp(-pow(d1, 2));
		consistency_score += consistency_score_temp;
	}
	consistency_score /= Voting_set.size();
	return consistency_score;
}
//Add corres among Match_up to Corres_sparse
int Group::RV_Corres_add(PointCloudPtr cloud_source, PointCloudPtr cloud_target, float mr, float RANSAC_inlier_judge_thresh, vector<Corres>& Match_up, vector<Corres>& voting_set, vector<Corres>& Corres_dense, int Corres_dense_thresh)
{
	int i, j;
	vector<int> target_idxs;
	//compute voting score for each correspondence
	for (i = 0; i < Match_up.size(); i++)
	{
		float vote_score = computePariwiseConsistency(cloud_source, cloud_target, mr, RANSAC_inlier_judge_thresh, Match_up[i], voting_set);
		Match_up[i].score = vote_score;
	}

	//IBI-module analysis
	///////**************Inverted Voting(IV)************/
	////for (i = 0; i < voting_set.size(); i++)
	////{
	////	float vote_score = computePariwiseConsistency(cloud_source, cloud_target, mr, RANSAC_inlier_judge_thresh, voting_set[i], Match_up);
	////	voting_set[i].score = vote_score;
	////}
	////sort(voting_set.begin(), voting_set.end(), [](Corres c1, Corres c2) {return c1.score > c2.score; });
	////for (i = 0; i < Match_up.size(); i++)
	////{
	////	for (j = 0; j < voting_set.size(); j++)
	////	{
	////		float consistency_score_temp2 = 0.0f;
	////		float x1 = cloud_source->points[Match_up[i].source_idx].x - cloud_source->points[voting_set[j].source_idx].x;
	////		float y1 = cloud_source->points[Match_up[i].source_idx].y - cloud_source->points[voting_set[j].source_idx].y;
	////		float z1 = cloud_source->points[Match_up[i].source_idx].z - cloud_source->points[voting_set[j].source_idx].z;
	////		float x2 = cloud_target->points[Match_up[i].target_idx].x - cloud_target->points[voting_set[j].target_idx].x;
	////		float y2 = cloud_target->points[Match_up[i].target_idx].y - cloud_target->points[voting_set[j].target_idx].y;
	////		float z2 = cloud_target->points[Match_up[i].target_idx].z - cloud_target->points[voting_set[j].target_idx].z;
	////		float dist1 = sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2));
	////		float dist2 = sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2));
	////		float d1 = abs(dist1 - dist2) / (10.0*mr);
	////		consistency_score_temp2 = exp(-pow(d1, 2)) / 2;
	////		if (consistency_score_temp2 < 1.5*mr) 
	////		{
	////			Corres_dense.push_back(Match_up[i]);
	////			break;
	////		}
	////	}
	////}
	
	//keep top Nrv as Corres_dense
	for (i = 0; i < Match_up.size(); i++) Corres_dense.push_back(Match_up[i]);
	sort(Corres_dense.begin(), Corres_dense.end(), [](Corres c1, Corres c2) {return c1.score > c2.score; });
	if (Corres_dense.size()> Corres_dense_thresh) Corres_dense.resize(Corres_dense_thresh);
	for (i = 0; i < Corres_dense.size(); i++)
	{
		target_idxs.push_back(Corres_dense[i].target_idx);
	}
	//label Corres_dense in Match_up(avoid selected again in following iteration)
	for (i = 0; i < Match_up.size(); i++)
	{
		if (find(target_idxs.begin(), target_idxs.end(), Match_up[i].target_idx) != target_idxs.end()) Match_up[i].flag = 1;
	}
	return Match_up.size();
}
//Compute OSTU thresh by scores
float Group::OSTU_thresh_GTM(vector<float>& values)
{
	int i, j;
	int Quant_num = 100;
	float score_sum, fore_score_sum = 0.0f;
	vector<int> score_Hist(Quant_num, 0);
	vector<float> score_sum_Hist(Quant_num, 0.0f);
	float max_score_value, min_score_value;
	vector<float> all_scores;
	for (i = 0; i < values.size(); i++)
	{
		score_sum += values[i];
		all_scores.push_back(values[i]);
	}
	sort(all_scores.begin(), all_scores.end());
	if (all_scores.size() == 0) max_score_value = all_scores[0];
	else max_score_value = all_scores[all_scores.size() - 1];
	min_score_value = all_scores[0];
	float Quant_step = (max_score_value - min_score_value) / Quant_num;
	for (i = 0; i < values.size(); i++)
	{
		int ID = values[i] / Quant_step;
		if (ID >= Quant_num) ID = Quant_num - 1;
		score_Hist[ID]++;
		score_sum_Hist[ID] += values[i];
	}
	float fmax = -1000;
	int n1 = 0, n2;
	float m1, m2, sb;
	float thresh = (max_score_value - min_score_value) / 2;//default value
	for (i = 0; i < Quant_num; i++)
	{
		float Thresh_temp = i * (max_score_value - min_score_value) / float(Quant_num);//Quant_step
		n1 += score_Hist[i];
		if (n1 == 0) continue;
		n2 = values.size() - n1;
		if (n2 == 0) break;
		fore_score_sum += score_sum_Hist[i];
		m1 = fore_score_sum / n1;
		m2 = (score_sum - fore_score_sum) / n2;
		sb = (float)n1*(float)n2*pow(m1 - m2, 2);
		if (sb > fmax)
		{
			fmax = sb;
			thresh = Thresh_temp;
		}
	}
	return thresh;
}

/**************GSAC************/
//compute top corres by sum of score
vector<Top3> Group::computeTopCorresByScore(vector<Corres>& Corres_added, vector<Top3>& top3, int RANSAC_iter)
{
	for (int i = 0; i < Corres_added.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			for (int k = 0; k < j; k++)
			{
				Top3 top;
				top.index1 = i;
				top.index2 = j;
				top.index3 = k;
				if (top.index1 >= Corres_added.size() || top.index2 >= Corres_added.size() || top.index3 >= Corres_added.size()) continue;
				top.score = Corres_added[i].score + Corres_added[j].score + Corres_added[k].score;
				top3.push_back(top);
			}
		}
	}
	sort(top3.begin(), top3.end(), [](Top3 t1, Top3 t2) {return t1.score > t2.score; });//Rank all the top3 in descending order of Score
	top3.resize(RANSAC_iter);
	for (int m = 0; m < top3.size(); m++)
	{
		Corres pair1;
		Corres pair2;
		Corres pair3;
		pair1.source_idx = Corres_added[top3[m].index1].source_idx;
		pair1.target_idx = Corres_added[top3[m].index1].target_idx;
		pair2.source_idx = Corres_added[top3[m].index2].source_idx;
		pair2.target_idx = Corres_added[top3[m].index2].target_idx;
		pair3.source_idx = Corres_added[top3[m].index3].source_idx;
		pair3.target_idx = Corres_added[top3[m].index3].target_idx;
		match1.push_back(pair1);
		match2.push_back(pair2);
		match3.push_back(pair3);
	}
	return top3;
}
//guided sample consensus
int Group::GSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Corres>& Match, float RANSAC_inlier_judge_thresh, int  _Iterations, vector<Top3>& top3, Eigen::Matrix4f& Mat)
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
	for (int i = 0; i < Iterations; i++)
	{
		int Match_Idx1, Match_Idx2, Match_Idx3;
		Match_Idx1 = top3[i].index1;
		Match_Idx2 = top3[i].index2;
		Match_Idx3 = top3[i].index3;
		if (top3[i].index1 >= Match.size() || top3[i].index2 >= Match.size() || top3[i].index3 >= Match.size()) continue;
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		point_s1 = cloud_source->points[Match[Match_Idx1].source_idx];
		point_s2 = cloud_source->points[Match[Match_Idx2].source_idx];
		point_s3 = cloud_source->points[Match[Match_Idx3].source_idx];
		point_t1 = cloud_target->points[Match[Match_Idx1].target_idx];
		point_t2 = cloud_target->points[Match[Match_Idx2].target_idx];
		point_t3 = cloud_target->points[Match[Match_Idx3].target_idx];

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
	Corres pair1;
	Corres pair2;
	Corres pair3;
	pair1.source_idx = Match[top3[x].index1].source_idx;
	pair1.target_idx = Match[top3[x].index1].target_idx;
	pair2.source_idx = Match[top3[x].index2].source_idx;
	pair2.target_idx = Match[top3[x].index2].target_idx;
	pair3.source_idx = Match[top3[x].index3].source_idx;
	pair3.target_idx = Match[top3[x].index3].target_idx;
	match1.push_back(pair1);
	match2.push_back(pair2);
	match3.push_back(pair3);
	return 1;
}

/**************validate globally************/
float Group::Overlap(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, float GTthresh, float overlap_thresh, Eigen::Matrix4f& Mat, vector<Corres>& Corres_added, vector<float>& overlaps, vector<Eigen::Matrix4f>& Mats, vector<vector<Corres>>& Corres_addeds)
{
	int i, j;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_src, *cloud_trans, Mat);
	int N = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>Idx;
	vector<float>Dist;
	kdtree.setInputCloud(cloud_tar);
	for (i = 0; i < cloud_trans->points.size(); i++)
	{
		int judge = kdtree.radiusSearch(cloud_trans->points[i], GTthresh, Idx, Dist);
		if (judge > 0)
			N++;
	}
	float overlap = float(N) / cloud_trans->points.size();
	////cout << "overlap=" << overlap << endl;
	if (overlap > overlap_thresh)
	{
		overlaps.push_back(overlap);
		Mats.push_back(Mat);
		Corres_addeds.push_back(Corres_added);
	}
	return overlap;
}

//IBI-module analysis
////HV: validate globally
int Group::Correct(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, float GTthresh, int correct_thresh, Eigen::Matrix4f& Mat, vector<Corres>& Corres_added, vector<int>& corrects, vector<Eigen::Matrix4f>& Mats)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Src(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Tar(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Trans(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointXYZ Point_Temp_s, Point_Temp_t;
	for (int i = 0; i < Corres_added.size(); i++)
	{
		Point_Temp_s = cloud_src->points[Corres_added[i].source_idx];
		Point_Temp_t = cloud_tar->points[Corres_added[i].target_idx];
		keyPointCloud_Src->push_back(Point_Temp_s);
		keyPointCloud_Tar->push_back(Point_Temp_t);
	}
	pcl::transformPointCloud(*keyPointCloud_Src, *keyPointCloud_Trans, Mat);
	int num = 0;
	for (int i = 0; i < Corres_added.size(); i++)
	{
		float x1 = keyPointCloud_Trans->points[i].x;
		float y1 = keyPointCloud_Trans->points[i].y;
		float z1 = keyPointCloud_Trans->points[i].z;
		float x2 = keyPointCloud_Tar->points[i].x;
		float y2 = keyPointCloud_Tar->points[i].y;
		float z2 = keyPointCloud_Tar->points[i].z;
		float dis = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
		if (dis <= GTthresh) num++;
	}
	if (num > correct_thresh)
	{
		corrects.push_back(num);
		Mats.push_back(Mat);
	}
	return num;
}

/**************Reset input************/
void Group::resetMatch(vector<Corres>& Match_up, vector<int>& target_idxs,vector<Corres>& Match, vector<vector<Corres>>& Instances, vector<Corres>& Instance)
{
	vector<Corres> Match_up_temp;
	vector<Corres> Match_temp;
	for (int i = 0; i < Match_up.size(); i++)
	{
		if (Match_up[i].flag != 1)
		{
			Match_up_temp.push_back(Match_up[i]);
			if (find(target_idxs.begin(), target_idxs.end(), i) != target_idxs.end())
			{
				Match_temp.push_back(Match_up[i]);
			}
		}
		else
		{
			if (find(target_idxs.begin(), target_idxs.end(), i) != target_idxs.end()) Instance.push_back(Match_up[i]);
		}
	}
	Instances.push_back(Instance);
	Match_up.clear();
	Match.clear();
	for (int i = 0; i < Match_up_temp.size(); i++)
	{
		Match_up.push_back(Match_up_temp[i]);
	}
	for (int i = 0; i < Match_temp.size(); i++)
	{
		Match.push_back(Match_temp[i]);
	}
}

//IBI-module analysis
////removeDuplicates
struct PointComparator {
	bool operator()(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) const {
		if (p1.x != p2.x) return p1.x < p2.x;
		if (p1.y != p2.y) return p1.y < p2.y;
		return p1.z < p2.z;
	}
};
void removeDuplicates(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	std::set<pcl::PointXYZ, PointComparator> uniquePoints;

	// Insert points into the set
	for (const auto& point : cloud->points) {
		uniquePoints.insert(point);
	}

	// Clear the original cloud and insert the unique points back
	cloud->points.clear();
	for (const auto& point : uniquePoints) {
		cloud->points.push_back(point);
	}

	// Update the width and height of the point cloud
	cloud->width = static_cast<uint32_t>(cloud->points.size());
	cloud->height = 1;
	cloud->is_dense = true;
}

/**************Evaluation************/
//Compute RRE, RTE
int Metrics(vector<Eigen::Matrix4f>& gts, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& Mats_hit, vector<REs>& RE, int& hit)
{
	int K = gts.size();
	int M = Mats.size();
	hit = 0;
	//compute F-norm distance
	Eigen::MatrixXf D(K, M);
	for (int k = 0; k < K; k++)
	{
		for (int m = 0; m < M; m++)
		{
			D(k, m) = (gts[k] - Mats[m]).norm();
		}
	}
	int S = min(K, M);
	int R = max(K, M);
	//assignment matrix
	Eigen::MatrixXf F(K, M);
	vector<int> ks, ms;
	for (int s = 0; s < S; s++)
	{
		// find minimum
		int k_min, m_min;
		D.minCoeff(&k_min, &m_min);
		F(k_min, m_min) = 1.0;
		// set D(k_min, :) and D(:, m_min) as maximum to avoid duplicate allocation.
		D.row(k_min).setConstant(numeric_limits<float>::max());
		D.col(m_min).setConstant(numeric_limits<float>::max());
	}
	// compute RRE and RTE
	float trace, RRE, RTE;
	for (int k = 0; k < K; k++)
	{
		for (int m = 0; m < M; m++)
		{
			if (F(k, m) == 1.0)
			{
				REs re;
				re.gt_id = k;
				re.mat_id = m;
				Eigen::Matrix3f R_mats = Mats[m].block<3, 3>(0, 0);
				Eigen::Vector3f t_mats = Mats[m].block<3, 1>(0, 3);
				Eigen::Matrix3f R_gts = gts[k].block<3, 3>(0, 0);
				Eigen::Vector3f t_gts = gts[k].block<3, 1>(0, 3);
				trace = (R_mats.transpose() * R_gts).trace();
				re.trace = trace;
				RRE = acos( (trace - 1) / 2 ) / M_PI * 180;
				re.RE = RRE;
				RTE = (t_gts - t_mats).norm();
				re.TE = RTE;
				// count whether the mat is 'hit'
				if (RRE < 20.0 && RTE < 0.5)
				{
					re.flag = 1;
					hit++;
					Mats_hit.push_back(Mats[m]);
				}
				RE.push_back(re);
			}
		}
	}
	return hit;
}