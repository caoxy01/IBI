#include "IBI_S2DC.h"
#include <stdarg.h>
extern bool add_overlap;
using namespace Eigen;
using namespace std;

/**************MAC************/
bool compare_vote_score(const Vote& v1, const Vote& v2) 
{
	return v1.score > v2.score;
}

bool compare_vote_degree(const Vote_exp& v1, const Vote_exp& v2) 
{
	return v1.degree > v2.degree;
}
double OTSU_thresh_MAC(Eigen::VectorXd values)
{
	int i;
	int Quant_num = 100;
	double score_sum = 0.0;
	double fore_score_sum = 0.0;
	vector<int> score_Hist(Quant_num, 0);
	vector<double> score_sum_Hist(Quant_num, 0.0);
	double max_score_value, min_score_value;
	vector<double> all_scores;
	for (i = 0; i < values.size(); i++)
	{
		score_sum += values[i];
		all_scores.push_back(values[i]);
	}
	sort(all_scores.begin(), all_scores.end());
	max_score_value = all_scores[all_scores.size() - 1];
	min_score_value = all_scores[0];
	double Quant_step = (max_score_value - min_score_value) / Quant_num;
	for (i = 0; i < values.size(); i++)
	{
		int ID = values[i] / Quant_step;
		if (ID >= Quant_num) ID = Quant_num - 1;
		score_Hist[ID]++;
		score_sum_Hist[ID] += values[i];
	}
	double fmax = -1000;
	int n1 = 0, n2;
	double m1, m2, sb;
	double thresh = (max_score_value - min_score_value) / 2;//default value
	for (i = 0; i < Quant_num; i++)
	{
		double Thresh_temp = i * (max_score_value - min_score_value) / double(Quant_num);
		n1 += score_Hist[i];
		if (n1 == 0) continue;
		n2 = values.size() - n1;
		if (n2 == 0) break;
		fore_score_sum += score_sum_Hist[i];
		m1 = fore_score_sum / n1;
		m2 = (score_sum - fore_score_sum) / n2;
		sb = (double)n1 * (double)n2 * pow(m1 - m2, 2);
		if (sb > fmax)
		{
			fmax = sb;
			thresh = Thresh_temp;
		}
	}
	return thresh;
}
void find_largest_clique_of_node(Eigen::MatrixXf& Graph, igraph_vector_ptr_t* cliques, vector<Corre_3DMatch>& correspondence, node_cliques* result, vector<int>& remain, int num_node, int est_num) {
	int* vis = new int[igraph_vector_ptr_size(cliques)];
	memset(vis, 0, igraph_vector_ptr_size(cliques));
#pragma omp parallel for
	for (int i = 0; i < num_node; i++)
	{
		result[i].clique_index = -1;
		result[i].clique_size = 0;
		result[i].clique_weight = 0;
		result[i].clique_num = 0;
	}

	for (int i = 0; i < remain.size(); i++)
	{
		igraph_vector_t* v = (igraph_vector_t*)VECTOR(*cliques)[remain[i]];
		float weight = 0;
		int length = igraph_vector_size(v);
		for (int j = 0; j < length; j++)
		{
			int a = (int)VECTOR(*v)[j];
			for (int k = j + 1; k < length; k++)
			{
				int b = (int)VECTOR(*v)[k];
				weight += Graph(a, b);
			}
		}
		for (int j = 0; j < length; j++)
		{
			int k = (int)VECTOR(*v)[j];
			if (result[k].clique_weight < weight)
			{
				result[k].clique_index = remain[i];
				vis[remain[i]]++;
				result[k].clique_size = length;
				result[k].clique_weight = weight;
			}
		}
	}

#pragma omp parallel for
	for (int i = 0; i < remain.size(); i++)
	{
		if (vis[remain[i]] == 0) {
			igraph_vector_t* v = (igraph_vector_t*)VECTOR(*cliques)[remain[i]];
			igraph_vector_destroy(v);
		}
	}

	vector<int>after_delete;
	for (int i = 0; i < num_node; i++)
	{
		if (result[i].clique_index < 0)
		{
			continue;
		}
		if (vis[result[i].clique_index] > 0)
		{
			vis[result[i].clique_index] = 0;
			after_delete.push_back(result[i].clique_index);
		}
		else if (vis[result[i].clique_index] == 0) {
			result[i].clique_index = -1;
		}
	}
	remain.clear();
	remain = after_delete;

	//reduce the number of cliques
	if (remain.size() > est_num)
	{
		vector<int>after_decline;
		vector<Vote>clique_score;
		for (int i = 0; i < num_node; i++)
		{
			if (result[i].clique_index < 0)
			{
				continue;
			}
			Vote t;
			t.index = result[i].clique_index;
			t.score = result[i].clique_weight;
			clique_score.push_back(t);
		}
		sort(clique_score.begin(), clique_score.end(), compare_vote_score);
		for (int i = 0; i < est_num; i++)
		{
			after_decline.push_back(clique_score[i].index);
		}
		remain.clear();
		remain = after_decline;
		clique_score.clear();
	}
	delete[] vis;
	return;
}
void weight_SVD(PointCloudPtr src_pts, PointCloudPtr des_pts, Eigen::VectorXd& weights, double weight_threshold, Eigen::Matrix4d& trans_Mat) 
{
	for (size_t i = 0; i < weights.size(); i++)
	{
		weights(i) = (weights(i) < weight_threshold) ? 0 : weights(i);
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weight;
	Eigen::VectorXd ones = weights;
	ones.setOnes();
	weight = (weights * ones.transpose());
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Identity = weight;

	Identity.setIdentity();
	weight = (weights * ones.transpose()).cwiseProduct(Identity);
	pcl::ConstCloudIterator<pcl::PointXYZ> src_it(*src_pts);
	pcl::ConstCloudIterator<pcl::PointXYZ> des_it(*des_pts);

	src_it.reset(); des_it.reset();
	Eigen::Matrix<double, 4, 1> centroid_src, centroid_des;
	pcl::compute3DCentroid(src_it, centroid_src);
	pcl::compute3DCentroid(des_it, centroid_des);

	src_it.reset(); des_it.reset();
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> src_demean, des_demean;
	pcl::demeanPointCloud(src_it, centroid_src, src_demean);
	pcl::demeanPointCloud(des_it, centroid_des, des_demean);

	Eigen::Matrix<double, 3, 3> H = (src_demean * weight * des_demean.transpose()).topLeftCorner(3, 3);

	// Compute the Singular Value Decomposition
	Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3> > svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<double, 3, 3> u = svd.matrixU();
	Eigen::Matrix<double, 3, 3> v = svd.matrixV();

	// Compute R = V * U'
	if (u.determinant() * v.determinant() < 0)
	{
		for (int x = 0; x < 3; ++x)
			v(x, 2) *= -1;
	}

	Eigen::Matrix<double, 3, 3> R = v * u.transpose();

	// Return the correct transformation
	Eigen::Matrix<double, 4, 4> Trans;
	Trans.setIdentity();
	Trans.topLeftCorner(3, 3) = R;
	const Eigen::Matrix<double, 3, 1> Rc(R * centroid_src.head(3));
	Trans.block(0, 3, 3, 1) = centroid_des.head(3) - Rc;
	trans_Mat = Trans;
}
double Distance(pcl::PointXYZ& A, pcl::PointXYZ& B) 
{
	double distance = 0;
	double d_x = (double)A.x - (double)B.x;
	double d_y = (double)A.y - (double)B.y;
	double d_z = (double)A.z - (double)B.z;
	distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
	return distance;
}
double evaluation_trans(vector<Corre_3DMatch>& Match, vector<Corre_3DMatch>& correspondnece, PointCloudPtr src_corr_pts, PointCloudPtr des_corr_pts, double weight_thresh, Eigen::Matrix4d& trans, double metric_thresh, float resolution, bool instance_equal) 
{

	PointCloudPtr src_pts(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr des_pts(new pcl::PointCloud<pcl::PointXYZ>);
	vector<double>weights;
	for (auto & i : Match)
	{
		if (i.score >= weight_thresh)
		{
			src_pts->push_back(i.src);
			des_pts->push_back(i.des);
			weights.push_back(i.score);
		}
	}
	if (weights.size() < 3)
	{
		return 0;
	}
	Eigen::VectorXd weight_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(weights.data(), weights.size());
	weights.clear();
	weights.shrink_to_fit();
	weight_vec /= weight_vec.maxCoeff();
	if (!add_overlap || instance_equal) {
		weight_vec.setOnes(); // 2023.2.23 
	}
	weight_SVD(src_pts, des_pts, weight_vec, 0, trans);
	PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*src_corr_pts, *src_trans, trans);
	//Eigen::Matrix4f trans_f = trans.cast<float>();
	//Eigen::Matrix3f R = trans_f.topLeftCorner(3, 3);
	double score = 0.0;
	int inlier = 0;
	int corr_num = src_corr_pts->points.size();
	for (int i = 0; i < corr_num; i++)
	{
		double dist = Distance(src_trans->points[i], des_corr_pts->points[i]);
		double w = 1;
		if (add_overlap)
		{
			w = correspondnece[i].score;
		}
		if (dist < metric_thresh)
		{
			inlier++;
			score += (metric_thresh - dist)*w / metric_thresh;
		}
	}
	src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
	des_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
	src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
	return score;
}


void eigenvector_centrality(Eigen::MatrixXd& Graph, Eigen::VectorXd& initial, Eigen::VectorXd& eigenvector) {
	eigenvector.resize(initial.size());
	eigenvector = initial;
	Eigen::VectorXd eigenvector_next = eigenvector;
	eigenvector_next.setZero();
	double max = 0;
	bool flag = false;

	Eigen::MatrixXd zero_one = Graph;
	for (int i = 0; i < Graph.rows(); i++)
	{
		for (int j = 0; j < Graph.cols(); j++) {
			zero_one(i, j) = Graph(i, j) ? 1 : 0;
		}
	}

	double tmp_max;
	while (!flag) {
		tmp_max = max;
		eigenvector_next = zero_one * eigenvector;
		//cout << eigenvector_next << endl;
		max = eigenvector_next.maxCoeff();
		cout << max << endl;
		for (int i = 0; i < eigenvector.size(); i++)
		{
			if (eigenvector(i) != eigenvector_next(i)) {
				break;
			}
			if (i == eigenvector.size() - 1)
			{
				flag = true;
			}
		}
		if (abs(1.0 / tmp_max - 1.0 / max) < 0.01)
		{
			flag = true;
		}
		eigenvector = eigenvector_next;
		eigenvector_next.setZero();
	}
	eigenvector /= max;
}
Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, float cmp_thresh) 
{
	int size = correspondence.size();
	Eigen::MatrixXf cmp_score;
	cmp_score.resize(size, size);
	cmp_score.setZero();
	Corre_3DMatch c1, c2;
	float score, src_dis, des_dis, dis, alpha_dis;

	for (int i = 0; i < size; i++)
	{
		c1 = correspondence[i];
		for (int j = i + 1; j < size; j++)
		{
			c2 = correspondence[j];
			src_dis = Distance(c1.src, c2.src);
			des_dis = Distance(c1.des, c2.des);
			dis = abs(src_dis - des_dis);
			alpha_dis = 10 * resolution;
			score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
			score = (score < cmp_thresh) ? 0 : score;
			cmp_score(i, j) = score;
			cmp_score(j, i) = score;
		}
	}
	if (sc2)
	{
		//Eigen::setNbThreads(6);
		cmp_score = cmp_score.cwiseProduct(cmp_score * cmp_score);
	}
	return cmp_score;
}

vector<int> vectors_intersection(vector<int> v1, vector<int> v2) {
	vector<int> v;
	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
	return v;
}
bool Group::MAC(PointCloudPtr src, PointCloudPtr des, vector<Corre_3DMatch>& correspondence, float resolution, float cmp_thresh, Eigen::Matrix4f& Mat)
{
	bool sc2 = true;
	bool GT_cmp_mode = false;
	int max_est_num = INT_MAX;
	string metric = "MAE";
	string descriptor = "NULL";
	string name = "test";
	int total_num = correspondence.size();

	Eigen::MatrixXf Graph = Graph_construction(correspondence, resolution, sc2, cmp_thresh);
	if (Graph.norm() == 0) return false;

	vector<int>degree(total_num, 0);
	vector<Vote_exp> pts_degree;
	for (int i = 0; i < total_num; i++)
	{
		Vote_exp t;
		t.true_num = 0;
		vector<int> corre_index;
		for (int j = 0; j < total_num; j++)
		{
			if (i != j && Graph(i, j)) {
				degree[i] ++;
				corre_index.push_back(j);
			}
		}
		t.index = i;
		t.degree = degree[i];
		t.corre_index = corre_index;
		pts_degree.push_back(t);
	}

	vector<Vote> cluster_factor;
	double sum_fenzi = 0;
	double sum_fenmu = 0;
	//omp_set_num_threads(12);
	for (int i = 0; i < total_num; i++)
	{
		Vote t;
		double sum_i = 0;
		double wijk = 0;
		int index_size = pts_degree[i].corre_index.size();
#pragma omp parallel
		{
#pragma omp for
			for (int j = 0; j < index_size; j++)
			{
				int a = pts_degree[i].corre_index[j];
				for (int k = j + 1; k < index_size; k++)
				{
					int b = pts_degree[i].corre_index[k];
					if (Graph(a, b)) {
#pragma omp critical
						wijk += pow(Graph(i, a) * Graph(i, b) * Graph(a, b), 1.0 / 3); //wij + wik
					}
				}
			}
		}

		if (degree[i] > 1)
		{
			double f1 = wijk;
			double f2 = degree[i] * (degree[i] - 1) * 0.5;
			sum_fenzi += f1;
			sum_fenmu += f2;
			double factor = f1 / f2;
			t.index = i;
			t.score = factor;
			cluster_factor.push_back(t);
		}
		else {
			t.index = i;
			t.score = 0;
			cluster_factor.push_back(t);
		}
	}
	double average_factor = 0;
	for (size_t i = 0; i < cluster_factor.size(); i++)
	{
		average_factor += cluster_factor[i].score;
	}
	average_factor /= cluster_factor.size();

	double total_factor = sum_fenzi / sum_fenmu;

	vector<Vote_exp> pts_degree_bac;
	vector<Vote>cluster_factor_bac;
	pts_degree_bac.assign(pts_degree.begin(), pts_degree.end());
	cluster_factor_bac.assign(cluster_factor.begin(), cluster_factor.end());

	sort(cluster_factor.begin(), cluster_factor.end(), compare_vote_score);
	sort(pts_degree.begin(), pts_degree.end(), compare_vote_degree);

	Eigen::VectorXd cluster_coefficients;
	cluster_coefficients.resize(cluster_factor.size());
	for (size_t i = 0; i < cluster_factor.size(); i++)
	{
		cluster_coefficients[i] = cluster_factor[i].score;
	}

	int cnt = 0;
	double OTSU = 0;
	if (cluster_factor[0].score != 0)
	{
		OTSU = OTSU_thresh_MAC(cluster_coefficients);
	}
	double cluster_threshold = min(OTSU, min(average_factor, total_factor));

	//cout << cluster_threshold << "->min(" << average_factor << " " << total_factor << " " << OTSU << ")" << endl;
	double weight_thresh = cluster_threshold;
	if (add_overlap)
	{
		weight_thresh = 0.5;
	}
	else {
		weight_thresh = 0;
	}

	//confidence score of corr
	if (!add_overlap)
	{
		for (size_t i = 0; i < total_num; i++)
		{
			correspondence[i].score = cluster_factor_bac[i].score;
		}
	}
	/*****************************************igraph**************************************************/
	igraph_t g;
	igraph_matrix_t g_mat;
	igraph_vector_t weights;
	igraph_vector_init(&weights, Graph.rows() * (Graph.cols() - 1) / 2);
	igraph_matrix_init(&g_mat, Graph.rows(), Graph.cols());

	//reduce the scale of graph
	if (cluster_threshold > 3 && correspondence.size() > 100 /*max(OTSU, total_factor) > 0.3*/) 
	{
		double f = 10;
		while (1)
		{
			if (f * max(OTSU, total_factor) > cluster_factor[99].score)
			{
				f -= 0.05;
			}
			else {
				break;
			}
		}
		for (int i = 0; i < Graph.rows(); i++)
		{
			if (cluster_factor_bac[i].score > f * max(OTSU, total_factor))
			{
				for (int j = i + 1; j < Graph.cols(); j++)
				{
					if (cluster_factor_bac[j].score > f * max(OTSU, total_factor))
					{
						MATRIX(g_mat, i, j) = Graph(i, j);
					}
				}
			}
		}
	}
	else {
		for (int i = 0; i < Graph.rows(); i++)
		{
			for (int j = i + 1; j < Graph.cols(); j++)
			{
				if (Graph(i, j))
				{
					MATRIX(g_mat, i, j) = Graph(i, j);
				}
			}
		}
	}

	igraph_set_attribute_table(&igraph_cattribute_table);
	igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED, 0, 1);
	const char* att = "weight";
	EANV(&g, att, &weights);

	//find all maximal cliques
	igraph_vector_ptr_t cliques;
	igraph_vector_ptr_init(&cliques, 0);

	igraph_maximal_cliques(&g, &cliques, 3, 0); 
	//igraph_largest_cliques(&g, &cliques);
	//print_and_destroy_cliques(&cliques);
	int clique_num = igraph_vector_ptr_size(&cliques);
	if (clique_num == 0) {
		cout << " NO CLIQUES! " << endl;
	}
	//cout << " clique computation: " << elapsed_time.count() << endl;

	//data cleansing
	igraph_destroy(&g);
	igraph_matrix_destroy(&g_mat);
	igraph_vector_destroy(&weights);

	vector<int>remain;
	for (int i = 0; i < clique_num; i++)
	{
		remain.push_back(i);
	}
	node_cliques* N_C = new node_cliques[(int)total_num];
	find_largest_clique_of_node(Graph, &cliques, correspondence, N_C, remain, total_num, max_est_num);

	PointCloudPtr src_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr des_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
	for (size_t i = 0; i < correspondence.size(); i++) {
		src_corr_pts->push_back(correspondence[i].src);
		des_corr_pts->push_back(correspondence[i].des);
	}

	/******************************************Registration***************************************************/
	double RE_thresh, TE_thresh, inlier_thresh;
	Eigen::Matrix4d best_est;
	inlier_thresh = 10 * resolution;
	bool found = false;
	double best_score = 0;
	vector<Corre_3DMatch>selected;
	vector<int>corre_index;
	int total_estimate = remain.size();
#pragma omp parallel for
	for (int i = 0; i < remain.size(); i++)
	{
		vector<Corre_3DMatch>Group;
		vector<int>selected_index;
		igraph_vector_t* v = (igraph_vector_t*)VECTOR(cliques)[remain[i]];
		int group_size = igraph_vector_size(v);
		for (int j = 0; j < group_size; j++)
		{
			Corre_3DMatch C = correspondence[VECTOR(*v)[j]];
			Group.push_back(C);
			selected_index.push_back(VECTOR(*v)[j]);
		}
		//igraph_vector_destroy(v);
		Eigen::Matrix4d est_trans;
		//score of cliques
		double score = evaluation_trans(Group, correspondence, src_corr_pts, des_corr_pts, weight_thresh, est_trans, inlier_thresh, resolution,true);
		//GT unkown
		if (score > 0)
		{
#pragma omp critical
			{
				if (best_score < score)
				{
					best_score = score;
					best_est = est_trans;
					selected = Group;
					corre_index = selected_index;
				}
			}
		}
	}
	Mat = best_est.cast<float>();
	//free memory
	igraph_vector_ptr_destroy(&cliques);
	//cout << total_estimate << " : " << clique_num << endl;

	correspondence.clear();
	correspondence.shrink_to_fit();
	degree.clear();
	degree.shrink_to_fit();
	pts_degree.clear();
	pts_degree.shrink_to_fit();
	pts_degree_bac.clear();
	pts_degree_bac.shrink_to_fit();
	cluster_factor.clear();
	cluster_factor.shrink_to_fit();
	cluster_factor_bac.clear();
	cluster_factor_bac.shrink_to_fit();
	delete[] N_C;
	remain.clear();
	remain.shrink_to_fit();
	selected.clear();
	selected.shrink_to_fit();
	corre_index.clear();
	corre_index.shrink_to_fit();
	src_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
	des_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
	return true;
}