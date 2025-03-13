#pragma once
#ifndef IBI_S2DC_IBI_S2DC_H
#define IBI_S2DC_IBI_S2DC_H
#define BOOST_TYPEOF_EMULATION
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <ctime>
#include <cmath>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_set>
//#include <boost/random.hpp>
#include <boost/thread/thread.hpp>
#include <Eigen/Core>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>        
#include <pcl/console/time.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h> 
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/normal_space.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
// igraph 0.9.9
#include <igraph.h>
//#define pcl_isfinite(x) std::isfinite(x)
#define Corres_view_gap -500
#define tR 30
#define tG 144
#define tB 255
#define sR 220
#define sG 20
#define sB 60
extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;

const double eps = 1.0e-6;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

typedef struct {
	float x;
	float y;
	float z;
}Vertex;
typedef struct {
	int pointID;
	Vertex x_axis;
	Vertex y_axis;
	Vertex z_axis;
}LRF;
typedef struct {
	int source_idx;
	int target_idx;
	LRF source_LRF;
	LRF target_LRF;
	float value;
	float score;
	int flag = 0;
	bool inlier=false;
}Corres;
typedef struct {
	int index1;
	int index2;
	int index3;
	float score;
}Top3;
typedef struct {
	int gt_id;
	int mat_id;
	float trace;
	float RE;
	float TE;
	int flag = 0;
}REs;
//MAC
typedef struct {
	int src_index;
	int des_index;
	pcl::PointXYZ src;
	pcl::PointXYZ des;
	Eigen::Vector3f src_norm;
	Eigen::Vector3f des_norm;
	Eigen::Matrix3f covariance_src, covariance_des;
	Eigen::Vector4f centeroid_src, centeroid_des;
	double score;
	int inlier_weight;
}Corre_3DMatch;
typedef struct {
	int index;
	double score;
}Vote;
typedef struct
{
	int index;
	int degree;
	double score;
	vector<int> corre_index;
	int true_num;
}Vote_exp;
typedef struct
{
	int clique_index;
	int clique_size;
	float clique_weight;
	int clique_num;
}node_cliques;
//SACCOT
typedef struct {
	int source_idx;
	int target_idx;
	int index;
	int degree;
}Node;
typedef struct {
	int index1;
	int index2;
	int index3;
	int degree;
	double compatibility;
	double area;
}COT;
typedef struct {
	int index1;
	int index2;
	int index3;
	int degree;
}Triplet;

void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx);
//generate random number
void Rand_x(int seed, int ds, vector<int>& output);
void Rand_3(int seed, int scale, int& output1, int& output2, int& output3);
//downsampling
void downSampling(vector<Corres>& Match_up, vector<Corres>& Match, int downSize, vector<int>& indices);
//compute resolution
double computeResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud);
//LRF
void TOLDI_LRF_for_cloud_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, vector<LRF>& Cloud_LRF);
void TOLDI_LRF_X_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex z_axis, float sup_radius, vector<float> PointDist, Vertex& x_axis);
void TOLDI_LRF_Y_axis(Vertex x_axis, Vertex z_axis, Vertex& y_axis);
void TOLDI_LRF_Z_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex& z_axis);
//label inliers
vector<bool> Label0(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Corres>& Match, float label_thresh, vector<Eigen::Matrix4f>& gts);
//generate corr with various outlier rates
pcl::PointCloud<pcl::PointXYZ> Syn_corr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, vector<Eigen::Matrix4f>& gts, float min, float max, vector<Corres>& Match_up);

class Registration
{
public:
	Registration() {};
	vector<Corres> Match;
	vector<Corres> match1;
	vector<Corres> match2;
	vector<Corres> match3;
	Eigen::Matrix4f matrix;
	vector<Top3> top3;
	vector<Node> nodes;
	vector<vector<double>> compatibility_matrix;
	vector<COT> cots;
	vector<Triplet> triplets;
};

class Group : public Registration
{
public:
	Group() {};
	//IBI_S2DC
	//GTM
    Eigen::MatrixXf M;//payoff matrix
    Eigen::VectorXf V;//population
	vector<Corres> Match_inlier;
	vector<Corres> Corres_sparse;
	void GTM_Corres_select(int Iterations, vector<Corres>& Match, vector<Corres>& Match_up, Eigen::MatrixXf& M, vector<Corres>& Match_inlier);
	void computepayoffMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar, vector<Corres>& Match, Eigen::MatrixXf& M);
	float OSTU_thresh_GTM(vector<float>& values);
	//RV
	vector<Corres> Voting_set;
	vector<Corres> Corres_dense;
	int RV_Corres_add(PointCloudPtr cloud_source, PointCloudPtr cloud_target, float mr, float RANSAC_inlier_judge_thresh, vector<Corres>& Match_up, vector<Corres>& voting_set, vector<Corres>& Corres_added, int Corres_added_thresh);
	float computePariwiseConsistency(PointCloudPtr cloud_source, PointCloudPtr cloud_target, float mr, float RANSAC_inlier_judge_thresh, Corres& c, vector<Corres>& Voting_set);
	//GSAC
	vector<Corres> match1;
	vector<Corres> match2;
	vector<Corres> match3;
	vector<Top3> computeTopCorresByScore(vector<Corres>& Corres_added, vector<Top3>& top3, int RANSAC_iter);
	void RANSAC_trans_est(pcl::PointXYZ& point_s1, pcl::PointXYZ& point_s2, pcl::PointXYZ& point_s3, pcl::PointXYZ& point_t1, pcl::PointXYZ& point_t2, pcl::PointXYZ& point_t3, Eigen::Matrix4f& Mat);
	double RANSAC_mae(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float inlier_threshold);
	int GSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Corres>& Match, float RANSAC_inlier_judge_thresh, int  _Iterations, vector<Top3>& top3, Eigen::Matrix4f& Mat);
	float Overlap(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, float GTthresh, float overlap_thresh, Eigen::Matrix4f& Mat, vector<Corres>& Corres_added, vector<float>& overlaps, vector<Eigen::Matrix4f>& Mats, vector<vector<Corres>>& Corres_addeds);
	int Correct(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, float GTthresh, int correct_thresh, Eigen::Matrix4f& Mat, vector<Corres>& Corres_added, vector<int>& corrects, vector<Eigen::Matrix4f>& Mats);
	int Inlier(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Corres>& Match, float inlier_thresh, Eigen::Matrix4f& gt);
	//reset input corres
	void resetMatch(vector<Corres>& Match_up, vector<int>& target_idxs, vector<Corres>& Match, vector<vector<Corres>>& Instances, vector<Corres>& Instance);
	
	//IBI-module analysis
	////RANSAC
	int RANSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Corres>& Match, float RANSAC_inlier_judge_thresh, int _Iterations, Eigen::Matrix4f& Mat);
	////MAC
	bool MAC(PointCloudPtr src, PointCloudPtr des, vector<Corre_3DMatch>& correspondence, float resolution, float cmp_thresh, Eigen::Matrix4f& Mat);
	////SACCOT
	void getNodeByMatch(vector<Corres>& Match);
	void computeMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar, vector<Corres>& Match, double compatibility_threshold);
	vector<COT> computeCOT(int topN);
	vector<Triplet> computeTriplet(int topN);
	int SAC_COT(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, std::vector<Corres>& Match, float RANSAC_inlier_judge_thresh, int _Iterations, Eigen::Matrix4f& Mat);
	////3DHough_Voting
	void HoughVoting_Corres_group(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Corres>& Match, vector<Corres> &Match_filteled, int Hough_bin_num);
};
////removeDuplicates
void removeDuplicates(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

//evaluate metrics
int Metrics(vector<Eigen::Matrix4f>& gts, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& Mats_hit, vector<REs>& RE, int& hit);

double Distance(pcl::PointXYZ& A, pcl::PointXYZ& B); 
void find_largest_clique_of_node(Eigen::MatrixXf& Graph, igraph_vector_ptr_t* cliques, vector<Corre_3DMatch>& correspondence, node_cliques* result, vector<int>& remain, int num_node, int est_num);
void weight_SVD(PointCloudPtr src_pts, PointCloudPtr des_pts, Eigen::VectorXd& weights, double weight_threshold, Eigen::Matrix4d& trans_Mat); 
double evaluation_trans(vector<Corre_3DMatch>& Match, vector<Corre_3DMatch>& correspondence, PointCloudPtr src_corr_pts, PointCloudPtr des_corr_pts, double weight_thresh, Eigen::Matrix4d& trans, double metric_thresh, float resolution, bool instance_equal);
double OTSU_thresh_MAC(Eigen::VectorXd values);
bool compare_vote_score(const Vote& v1, const Vote& v2);
bool compare_vote_degree(const Vote_exp& v1, const Vote_exp& v2);
Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, float cmp_thresh);

double computeDistance(pcl::PointXYZ &point_i, pcl::PointXYZ &point_j);
float computeRigidity(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j);
double computeCompatibility(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j);

//visualization
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing);
////synthetic
void visualization1(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f Mat, vector<Corres>& Corres, float resolution);
//////////groundtruths
void visualization01(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& GTs);
//////////mat n one iteration
void visualization31(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f& Mat);
//////////instances
void visualization4(int scene_num, PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<vector<Corres>> Instances, float resolution);
////real
//////////groundtruths
void visualization02(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& GTs);
//////////mat n one iteration
void visualization32(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, Eigen::Matrix4f& Mat);
//////////Match_up w. inliers
void visualization7(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar_vis, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<Corres>& Corres, float resolution);
//////////RV topNrv: allgreen
void visualization9(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<Corres>& Corres, float resolution);
//////////GSAC
void visualization8(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar_vis, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<Corres>& match1, vector<Corres>& match2, vector<Corres>& match3, float resolution);
//////////instances
//////////OBB
void visualization5(int scene_num, PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<vector<Corres>> Instances, float resolution);
//////////AABB
void visualization6(int scene_num, PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<vector<Corres>> Instances, float resolution);
#endif //IBI_S2DC_IBI_S2DC_H
