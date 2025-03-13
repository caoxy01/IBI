#include <vector>
#include <string>
#include <sstream>
#include <ostream>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <time.h>
#include "IBI_S2DC.h"
bool add_overlap;
using namespace std;

int main()
{
	string dataload, dataPath, resultPath, synPath;
	//dataPath = "C:\\Datasets\\IBI\\Synthetic";
	////resultPath = "C:\\Result\\IBI\\Synthetic";
	//int num = 623;//synthetic

	//dataPath = "C:\\Datasets\\IBI\\Synthetic";//Synthetic
	//synPath = "C:\\Datasets\\IBI\\Oulier_ratio\\0.98-0.99";//IBI Synthetic
	//resultPath = "C:\\Result\\IBI\\Outlier_ratio\\0.98-0.99";
	//int num = 623;//outlier_ratio
	//float min = 0.98f;
	//float max = 0.99f;

	dataPath = "C:\\Datasets\\IBI\\Real";
	//resultPath = "C:\\Result\\IBI\\Real";
	int num = 173;//real
	
	/////////////////////////
	pcl::console::TicToc time;
	time.tic();
	//ofstream s2dc_fw(resultPath + "\\s2dc.txt");
	int select_thresh = 5;
	int downSize = 1024;
	int Ngtm = 20;
	int Nrv = 300;
	int Ngsac = 100;

	/////////////////////////
	int correct_thresh = 100;
	float overlap_thresh;
	float gt_thresh;
	float Threshold_Radius;
	if (num == 623)
	{
		overlap_thresh = 0.85f;
		gt_thresh = 1.5f;
		Threshold_Radius = 10.0f;
	}
	if (num == 173)
	{
		overlap_thresh = 0.7f;
		gt_thresh = 3.0f;
		Threshold_Radius = 1.0f;
	}
	vector<REs> RE;
	vector<REs> ecc_RE;
	vector<REs> pointclm_RE;

	int y = 0;
	int Match_up_inlier=0;
	int Is;
	int ecc_Is;
	int pointclm_Is;
	float MHRs = 0.0f;
	float MHPs = 0.0f;
	float MHF1s = 0.0f;
	float Costs = 0.0f;
	vector<string> pairs;
	for (int i = 0; i < num; i++) pairs.push_back(to_string(i));
	for (int p = 0; p < pairs.size(); p++)
	{ 
		//if (p < 156)continue; //Fig. 1(real)
		//if (p < 11)continue; //Fig. 3+4(real)
		//if (p < 3)continue; //Fig. 5(a)-1(synthetic)
		//if (p < 9)continue; //Fig. 5(a)-2(real)
		//if (p < 22)continue; //Fig. 5(a)-3(real)
		//if (p < 7)continue; //Fig. 5(b)-1(synthetic)
		//if (p < 27)continue; //Fig. 5(b)-2(real)
		//if (p < 15)continue; //Fig. 5(b)-3(real)
		//if (p < 540)continue; //Fig. 6(synthetic)
		//if (p < 111)continue; //Fig. 7(real)
		////if (p < 7)continue; //Fig. 11-1(synthetic)
		//if (p < 26)continue; //Fig. 11-2(synthetic)
		//if (p < 47)continue; //Fig. 11-3(synthetic)
		//if (p < 11)continue; //Fig. 11-4(synthetic)
		//if (p < 3)continue; //Fig. 20-4(real)
		//if (p < 48)continue; //Fig. 20-5(real)
		//if (p < 156)continue; //Fig. 20-6(real)
		cout << p << "/ ";
		string filename = pairs[p];
		string corr_path, gtmat_path, label_path, src_path, tar_path, ecc_path, ecc_Is_path, pointclm_path;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar(new pcl::PointCloud<pcl::PointXYZ>); 
		pcl::PointCloud<pcl::PointXYZ> cloud_source;
		pcl::PointCloud<pcl::PointXYZ> cloud_target;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
		if (num == 623) //Synthetic
		{
			corr_path = dataPath + "\\" + filename + "\\corr.txt";
			gtmat_path = dataPath + "\\" + filename + "\\gt.txt";
			src_path = dataPath + "\\" + filename + "\\src.pcd";
			tar_path = dataPath + "\\" + filename + "\\tgt.pcd";
		}
		if (num == 173) //Real
		{
			corr_path = dataPath + "\\" + filename + "\\corr.txt";
			gtmat_path = dataPath + "\\" + filename + "\\gt.txt";
			src_path = dataPath + "\\" + filename + "\\tgt.pcd";
			tar_path = dataPath + "\\" + filename + "\\src.pcd"; 
			string tar_vis_path = dataPath + "\\" + filename + "\\src_vis.pcd";
			pcl::io::loadPCDFile<pcl::PointXYZRGB>(tar_vis_path, *cloud_tar_vis);
		}
		clock_t begin, end;
		double  cost;

		ofstream metrics_fw(resultPath + "\\" + filename + ".txt", ios_base::out);
		/////////////////////////load pointcloud
		pcl::io::loadPCDFile<pcl::PointXYZ>(src_path, *cloud_src);
		pcl::io::loadPCDFile<pcl::PointXYZ>(tar_path, *cloud_tar); 
		/////////////////////////read GTmat and save as vector gts
		ifstream gtmat_file(gtmat_path);
		string gtmat_line;
		vector<Eigen::Matrix4f> gts;
		while (getline(gtmat_file, gtmat_line))
		{
			Eigen::Matrix4f gt;
			Eigen::Matrix4f gt_iv;
			sscanf(gtmat_line.c_str(), "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
				&gt(0, 0), &gt(0, 1), &gt(0, 2), &gt(0, 3),
				&gt(1, 0), &gt(1, 1), &gt(1, 2), &gt(1, 3),
				&gt(2, 0), &gt(2, 1), &gt(2, 2), &gt(2, 3),
				&gt(3, 0), &gt(3, 1), &gt(3, 2), &gt(3, 3));

				gts.push_back(gt);
		}
		gtmat_file.close();


		/////////////////////////visualize the ground trouth transformations
		//visualization01(cloud_src, cloud_tar, gts);
		//visualization02(cloud_src, cloud_tar_vis2, gts);

		///////////////////////////synthetic dataset with different outlier_ratio
		//ofstream corr_fw(synPath + "\\" + filename + "\\corr.txt", ios_base::out);
		//cloud_source = *cloud_src;
		//cloud_target = Syn_corr(cloud_src, gts, min, max, Match_up);
		//pcl::io::savePCDFileASCII(synPath + "\\" + filename + "\\tgt.pcd", cloud_target);
		//pcl::io::savePCDFileASCII(synPath + "\\" + filename + "\\src.pcd", cloud_source);
		//cloud_tar = cloud_target.makeShared();
		//for (int c = 0; c < Match_up.size(); c++) corr_fw << Match_up[c].source_idx << " " << Match_up[c].target_idx << endl;
		//corr_fw.close();

		///////////////////////////read corr
		ifstream corr_file(corr_path);
		vector<Corres> Match_up;
		string corr_line;
		while (getline(corr_file, corr_line))
		{
			istringstream iss(corr_line);
			Corres corr;
			if (num == 623) //Synthetic
			{
				if (!(iss >> corr.source_idx >> corr.target_idx))
				{
					cerr << "Failed to read line from corr.txt!" << endl;
					return 1;
				}
				Match_up.push_back(corr);
			}
			if (num == 173) //Real
			{
				if (!(iss >> corr.target_idx >> corr.source_idx))
				{
					cerr << "Failed to read line from corr.txt!" << endl;
					return 1;
				}
				Match_up.push_back(corr);
			}
		}
		corr_file.close();

		/////////////////////////set label &count inliers
		//vector<bool> labels = Label0(cloud_src, cloud_tar, Match_up, resolution, gts);
		//for (int i = 0; i < Match_up.size(); i++)
		//{
		//	Match_up[m].inlier = labels[m];
		//  if (Match_up[i].inlier) Match_up_inlier++;
		//}
		//cout << " Match_up_inlier = " << Match_up_inlier << endl;
		///////////////////////////set score 
		//for (int m = 0; m < Match_up.size(); m++) Match_up[m].score = scores[m];
		
		/////////////////////////compute resolution
		float resolution;
		float res_src = computeResolution(cloud_src);
		float res_tar = computeResolution(cloud_tar);
		resolution = (res_src + res_tar) / 2;
		
		/////////////////////////downsampling
		vector<Corres> Match;
		vector<int> target_idxs;
		if (downSize < Match_up.size()) 
		{
			downSampling(Match_up, Match, downSize, target_idxs);
		}
		else
		{
			for (int m = 0; m < Match_up.size(); m++) 
			{
				Match.push_back(Match_up[m]);
				target_idxs.push_back(Match_up[m].target_idx);
			}
		}

		vector<Eigen::Matrix4f> Mats;
		vector<float> overlaps;
		vector<int> corrects;

		int x = 1;
		int select = 100;
		int correct = 250;
		float overlap = 2.0f;
		float GTthresh = gt_thresh * resolution;
		float GSAC_inlier_judge_thresh = Threshold_Radius * resolution;
		//int initial_inlier = 0;
		//int GTM_inlier=0;
		//int consistency_inlier = 0;
		//int RV_inlier=0;

		float MHR = 0.0f;
		float MHP = 0.0f;
		float MHF1 = 0.0f;
		
		begin = clock();
		metrics_fw << "pair" << p + 1 << "  Match_up = " << Match_up.size() << "  resolution = " << resolution << "  cloud_src = " << cloud_src->points.size() << "  cloud_tar = " << cloud_tar->points.size() << endl;
		vector<vector<Corres>> Instances;
		vector<Corres> Instance;
		vector<vector<Corres>> Corres_denses;

		//------------------------------------------------------------------------------------------------//
		while (1)
		{
			//cout << "---iteration " << x << "---" << endl;
			/////////////////////////perform IBI during each scene
			Registration reg;
			Group group;

			vector<Corres> Corres_sparse;
			Eigen::MatrixXf M(Match.size(), Match.size());
			/////////////////////////select Corres by GTM
			group.computepayoffMatrix(cloud_src, cloud_tar, Match, M);
			group.GTM_Corres_select(Ngtm, Match, Match_up, M, Corres_sparse);
			group.Corres_sparse = Corres_sparse;
			int select = Corres_sparse.size();
			if (select <= select_thresh)break;

			vector<Corres> Corres_dense;
			/////////////////////////add Corres by RV
			group.RV_Corres_add(cloud_src, cloud_tar, resolution, GSAC_inlier_judge_thresh, Match_up, Corres_sparse, Corres_dense, Nrv);
			group.Corres_dense = Corres_dense;

			Eigen::Matrix4f Mat;
			vector<Top3> top3;
			/////////////////////////guided sample consensus by Corres_dense
			top3 = group.computeTopCorresByScore(Corres_dense, top3, Ngsac);
			group.GSAC(cloud_src, cloud_tar, Corres_dense, GSAC_inlier_judge_thresh, Ngsac, top3, Mat);
			
			//IBI-module analysis
			////RANSAC
			//////group.RANSAC(cloud_src, cloud_tar, Corres_dense, GSAC_inlier_judge_thresh, RANSAC_iter, Mat);
			////SACCOT
			////group.getNodeByMatch(Corres_dense);
			////group.computeMatrix(cloud_src, cloud_tar, Corres_dense, 0.97);
			////vector<COT> cots = group.computeCOT(RANSAC_iter);
			////vector<Triplet> triplets = group.computeTriplet(RANSAC_iter);
			vector<Corres> match1 = group.match1;
			vector<Corres> match2 = group.match2;
			vector<Corres> match3 = group.match3;
			////group.SAC_COT(cloud_src, cloud_tar, Corres_dense, GSAC_inlier_judge_thresh, RANSAC_iter, Mat);
			////MAC
			//////add_overlap = false;
			//////vector<Corre_3DMatch> Corres_dense_3DMatch;
			//////for (int a = 0; a < Corres_dense.size(); a++) 
			//////{
			//////	Corre_3DMatch corr_3d;
			//////	corr_3d.src_index = Corres_dense[a].source_idx;
			//////	corr_3d.des_index = Corres_dense[a].target_idx;
			//////	corr_3d.src = cloud_src->points[Corres_dense[a].source_idx];
			//////	corr_3d.des = cloud_src->points[Corres_dense[a].target_idx];
			//////	Corres_dense_3DMatch.push_back(corr_3d);
			//////}
			//////group.MAC(cloud_src, cloud_tar, Corres_dense_3DMatch, resolution, 0.99, Mat);


			/////////////////////////validate globally
			overlap = group.Overlap(cloud_src, cloud_tar, GTthresh, overlap_thresh, Mat, Corres_dense, overlaps, Mats, Corres_denses);
			//IBI-module analysis
			////correct = group.Correct(cloud_src, cloud_tar, GTthresh, correct_thresh, Mat, Corres_dense, corrects, Mats);

			int match_size = Match.size();
			/////////////////////////reset the input correspondence set
			group.resetMatch(Match_up, target_idxs, Match, Instances, Instance);

			if (Match.size() <= 5)break;
			/////////////////////////visualization(Synthetic)
			//visualization1(cloud_src, cloud_tar, Mat, Match_up, resolution);
			//visualization1(cloud_src, cloud_tar, Mat, Match, resolution);
			//visualization1(cloud_src, cloud_tar, Mat, Corres_sparse, resolution);
			//visualization1(cloud_src, cloud_tar, Mat, Corres_dense, resolution);

			//metrics_fw << "instance # " << x << endl;
			//metrics_fw << "Match = " << match_size;
			//metrics_fw << "  Corres_sparse = " << Corres_sparse.size();
			//metrics_fw << "  Corres_dense = " << Corres_dense.size() << endl;
			x++;

			Corres_sparse.clear();
			Corres_dense.clear();
			group.Corres_sparse.clear();
			group.Corres_dense.clear();
		}

		end = clock();
		cost = (double)(end - begin) / CLOCKS_PER_SEC;
		
		/////////////////////////evaluate the method
		int K = gts.size();

		//IBI-module analysis
		////merge duplicated registrations
		//Eigen::Matrix4f I;
		//I.setIdentity();
		//for (int m = 0; m < Mats.size(); m++)
		//{
		//	for (int n = m + 1; n < Mats.size(); )
		//	{
		//		float M_error1 = (Mats[m].inverse()*Mats[n] - I).norm();
		//		//float M_error2 = (Mats[m] - Mats[n]).norm();
		//		//cout << M_error2 << " ";
		//		if (M_error1 < 2.0) Mats.erase(Mats.begin()+n);//synthetic 2.0, real 0.7
		//		else n++;
		//	}
		//}

		int M = Mats.size();
		vector<Eigen::Matrix4f> Mats_hit;
		Metrics(gts, Mats, Mats_hit, RE, Is);
		
		//visualization4(p + 1, cloud_src, cloud_tar, Mats_hit, gts, Instances, resolution); //Synthetic
		//visualization6(p + 1, cloud_src, cloud_tar_vis, Mats, gts, Instances, resolution); //Real

		//vector<Corres> Match_final;
		//for (int m = 0; m < Corres_addeds.size(); m++)
		//	for (int n = 0; n < Corres_addeds[m].size(); n++)Match_final.push_back(Corres_addeds[m][n]);
		//visualization7(cloud_src, cloud_tar_vis, Mats, gts, Match_final, resolution);
		//cout << "Match_final = "<< Match_final.size() <<endl;

		MHR = (float)Is / K;
		//set MHP = 0 when Mats.size() = 0
		if (M == 0) MHP = 0.0f;
		else MHP = (float)Is / M;
		MHF1 = 2 * MHR * MHP / (MHR + MHP + 1e-12);

		/////////////////////////add all num up
		MHRs += MHR;
		MHPs += MHP;
		MHF1s += MHF1;
		float Cost = cost;
		Costs += Cost;
		//s2dc_fw << p << " : " << MHR << " " << Is << " " << K << endl;

		metrics_fw << "K = " << gts.size() << " M = " << Mats.size() << " Is = " << Is << endl;
		cout << " Is = " << Is << "  /K = " << gts.size() << " M = " << Mats.size();
		metrics_fw << "cost = " << cost << " MHR = " << MHR << " MHP = " << MHP << " MHF1 = " << MHF1 << endl;
		cout << " MHR = " << MHR << " MHP = " << MHP << " MHF1 = " << MHF1 << endl;
		metrics_fw << "mean_MHR = " << MHRs / (y + 1) << ", mean_MHP = " << MHPs / (y + 1) << ", mean_MHF1 = " << MHF1s / (y + 1) << ", mean_Cost = " << Costs / (y + 1) << endl;
		cout << "  " << cost << endl;
		y++;
		//record the linear assign similarity
		for (int r = 0; r < RE.size(); r++)
		{
			if (RE[r].flag != 0) metrics_fw << " hit:";
			metrics_fw << " # " << RE[r].mat_id << " - " << RE[r].gt_id << "  trace = " << RE[r].trace << "  RRE = " << RE[r].RE << "  RTE = " << RE[r].TE << endl;
		}
		metrics_fw.close();
		RE.clear();
	}
	///////////////////////evaluate the method
	float mean_MHR = MHRs / num;
	float mean_MHP = MHPs / num;
	float mean_MHF1 = MHF1s / num;
	//s2dc_fw.close();

	ofstream totaltime_fw(resultPath + "\\metrics.txt");
	totaltime_fw << resultPath << endl;
	if (num == 623) totaltime_fw << "downsample"<< downSize <<" synthetic" << endl;
	if (num == 173) totaltime_fw << "downsample" << downSize << " real" << endl;

	totaltime_fw << "Ngtm = " << Ngtm << ",  Nrv = " << Nrv << ",  Ngsac = " << Ngsac << endl;
	totaltime_fw << "overlap_thresh = " << overlap_thresh << "  GTthresh = " << gt_thresh << " * resolution" << "  GSAC_inlier_judge_thresh = " << Threshold_Radius << " * resolution" << endl;
	totaltime_fw << "mean_time = " << time.toc() / 1000 / num << " s, ";
	totaltime_fw << "MHR = " << mean_MHR << ", MHP = " << mean_MHP << ", MHF1 = " << mean_MHF1 << endl;
	totaltime_fw.close();
	cout << "MHR = " << mean_MHR << ", MHP = " << mean_MHP << ", MHF1 = " << mean_MHF1 << ", Cost = " << time.toc() / 1000 / num << " s" << endl;
	return 0;
}