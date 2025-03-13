#include "IBI_S2DC.h"

bool next_iteration = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown())
		next_iteration = true;
}

void visualization01(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& GTs)
{
	pcl::visualization::PCLVisualizer viewer("GTs");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 1, 165, 175);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 128, 128, 128);
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
	viewer.setBackgroundColor(255, 255, 255);
	//viewer.setSize(3072, 1920);
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src" + to_string(g));
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//viewer.addPointCloud<pcl::PointXYZ>(cloud_trans_src, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_trans_src, 0, 255, 0), "sample cloud");
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
				//----------------------------------------------------------------------------------------------//
				viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 0.0, 1.0, 0.0, "AABB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB" + to_string(g));//图形的不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "AABB" + to_string(g));//线宽
				Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector" + to_string(g));
				//viewer.addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector" + to_string(g));
				//viewer.addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector" + to_string(g));

			}
		}
		next_iteration = false;
	}
}
void visualization02(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& GTs)
{
	pcl::visualization::PCLVisualizer viewer("GTs");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 1,165,175);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tar);
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, rgb, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
	viewer.setBackgroundColor(255, 255, 255);
	//viewer.setSize(3072, 1920);
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				//viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src" + to_string(g));
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//viewer.addPointCloud<pcl::PointXYZ>(cloud_trans_src, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_trans_src, 0, 255, 0), "sample cloud");
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
				//----------------------------------------------------------------------------------------------//
				viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB" + to_string(g));//图形的不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "AABB" + to_string(g));//线宽
				Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector" + to_string(g));
				//viewer.addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector" + to_string(g));
				//viewer.addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector" + to_string(g));

			}
		}
		next_iteration = false;
	}
}
void visualization1(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f Mat, vector<Corres>& Corres, float resolution)
{
	//visulization
	pcl::visualization::PCLVisualizer viewer("GTM_CV");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 0, 166, 237);//blue
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 128, 128, 128);//gray
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 181, 181, 181);
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 139, 136, 120);

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y -= 100 * resolution;
	}
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");

	double line_R = 0;
	double line_G = 0;
	double line_B = 0;
	double max_channel = 255;
	for (size_t i = 0; i < Corres.size(); ++i)
	{
		if (Corres[i].inlier)
		{
			line_G = 255;
		}
		else
		{
			line_R = 255;
		}
		line_R /= max_channel;
		line_G /= max_channel;
		line_B /= max_channel;
		int idx1 = Corres[i].source_idx;
		int idx2 = Corres[i].target_idx;
		std::stringstream SS_line_b1;
		SS_line_b1 << "line" << i;
		viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(cloud_src->points[idx1], cloud_tar->points[idx2], line_R, line_G, line_B, SS_line_b1.str());
		//if (Corres[i].inlier) viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, SS_line_b1.str());//不透明度
		//else viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, SS_line_b1.str());//不透明度
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, SS_line_b1.str());//点大小
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, SS_line_b1.str());//线宽	
	}

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y += 100 * resolution;
	}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);

	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src, Mat);
			viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src");
			viewer.removePointCloud("keyPoint_src");
			viewer.removePointCloud("keyPoint_tar");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src");
			//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
		}
		next_iteration = false;
	}
	//system("pause");
}

void visualization31(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f& Mat)
{
	pcl::visualization::PCLVisualizer viewer("Mat");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 255, 16, 13);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 1, 165, 175);
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
	viewer.setBackgroundColor(255, 255, 255);
	//viewer.setSize(3072, 1920);
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src, Mat);
			viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src");
		}
		next_iteration = false;
	}
}
void visualization32(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, Eigen::Matrix4f& Mat)
{
	pcl::visualization::PCLVisualizer viewer("Mat");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 1, 165, 175);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tar);
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, rgb, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
	viewer.setBackgroundColor(255, 255, 255);
	//viewer.setSize(3072, 1920);
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src, Mat);
			viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src");
		}
		next_iteration = false;
	}
}
void visualization4(int scene_num, PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<vector<Corres>> Instances, float resolution)
{
	//visulization
	pcl::visualization::PCLVisualizer viewer("GTM_CV");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 0, 166, 237);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 128, 128, 128);
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 181, 181, 181);
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 139, 136, 120);

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y -= 100 * resolution;
	}
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");

	for (int i = 0; i < Mats.size(); i++) 
	{
		double line_R = (rand() % 255);
		double line_G = (rand() % 255);
		double line_B = (rand() % 255);
		double max_channel = std::max(line_R, std::max(line_G, line_B));
		line_R /= max_channel;
		line_G /= max_channel;
		line_B /= max_channel;
		for (size_t j = 0; j < Instances[i].size(); ++j)
		{
			int idx1 = Instances[i][j].source_idx;
			int idx2 = Instances[i][j].target_idx;
			//        cout << "m = " << j << endl;
			//        cout << "match1.source_idx = " << idx1 << "  match1.target_idx = " << idx2 << endl;
			//        cout << "match2.source_idx = " << idx3 << "  match2.target_idx = " << idx4 << endl;
			//        cout << "match3.source_idx = " << idx5 << "  match3.target_idx = " << idx6 << endl;
			std::stringstream SS_line_b1;
			SS_line_b1 << "line" << i<<"_"<<j;
			viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(cloud_src->points[idx1], cloud_tar->points[idx2], line_R, line_G, line_B, SS_line_b1.str());
		}
	}
	
	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y += 100 * resolution;
	}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);


	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src, 0, 255, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				//viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src" + to_string(g));
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
				Eigen::Quaternionf quat(rotational_matrix_OBB);
				viewer.addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "OBB" + to_string(g));//颜色
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "OBB" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "OBB" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector" + to_string(g));
				//viewer.addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector" + to_string(g));
				//viewer.addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector" + to_string(g));

			}
			for (int g = 0; g < Mats.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src2, Mats[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src2, 255, 0, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src2);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				viewer.addPointCloud(cloud_trans_src2, cloud_color_handler_src, "cloud_trans_source2" + to_string(scene_num) + "_" + to_string(g));
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_source2" + to_string(scene_num) + "_" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
				Eigen::Quaternionf quat(rotational_matrix_OBB);
				viewer.addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB2" + to_string(scene_num) + "_" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "OBB2" + to_string(scene_num) + "_" + to_string(g));//颜色
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "OBB2" + to_string(scene_num) + "_" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "OBB2" + to_string(scene_num) + "_" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector2" + to_string(scene_num) + "_" + to_string(g));
				//viewer.addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector2" + to_string(scene_num) + "_" + to_string(g));
				//viewer.addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector2" + to_string(scene_num) + "_" + to_string(g));
			}
		}
		next_iteration = false;
	}
	//system("pause");
}
void visualization5(int scene_num, PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<vector<Corres>> Instances, float resolution)
{
	//visulization
	pcl::visualization::PCLVisualizer viewer("GTM_CV");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 0, 166, 237);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tar);
	//pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 128, 128, 128);
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 181, 181, 181);
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 139, 136, 120);

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y -= 100 * resolution;
	}
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, rgb, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");

	for (int i = 0; i < Mats.size(); i++)
	{
		double line_R = (rand() % 255);
		double line_G = (rand() % 255);
		double line_B = (rand() % 255);
		double max_channel = std::max(line_R, std::max(line_G, line_B));
		line_R /= max_channel;
		line_G /= max_channel;
		line_B /= max_channel;
		for (size_t j = 0; j < Instances[i].size(); ++j)
		{
			int idx1 = Instances[i][j].source_idx;
			int idx2 = Instances[i][j].target_idx;
			//        cout << "m = " << j << endl;
			//        cout << "match1.source_idx = " << idx1 << "  match1.target_idx = " << idx2 << endl;
			//        cout << "match2.source_idx = " << idx3 << "  match2.target_idx = " << idx4 << endl;
			//        cout << "match3.source_idx = " << idx5 << "  match3.target_idx = " << idx6 << endl;
			std::stringstream SS_line_b1;
			SS_line_b1 << "line" << i << "_" << j;
			viewer.addLine< pcl::PointXYZ, pcl::PointXYZRGB>(cloud_src->points[idx1], cloud_tar->points[idx2], line_R, line_G, line_B, SS_line_b1.str());
		}
	}

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y += 100 * resolution;
	}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);


	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src, 0, 255, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src" + to_string(g));
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
				Eigen::Quaternionf quat(rotational_matrix_OBB);
				viewer.addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "OBB" + to_string(g));//颜色
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "OBB" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "OBB" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector" + to_string(g));
				//viewer.addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector" + to_string(g));
				//viewer.addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector" + to_string(g));

			}
			for (int g = 0; g < Mats.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src2, Mats[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src2, 255, 0, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src2);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				viewer.addPointCloud(cloud_trans_src2, cloud_color_handler_src, "cloud_trans_source2" + to_string(scene_num) + "_" + to_string(g));
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_source2" + to_string(scene_num) + "_" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
				Eigen::Quaternionf quat(rotational_matrix_OBB);
				viewer.addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB2" + to_string(scene_num) + "_" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "OBB2" + to_string(scene_num) + "_" + to_string(g));//颜色
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.6, "OBB2" + to_string(scene_num) + "_" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "OBB2" + to_string(scene_num) + "_" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector2" + to_string(scene_num) + "_" + to_string(g));
				//viewer.addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector2" + to_string(scene_num) + "_" + to_string(g));
				//viewer.addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector2" + to_string(scene_num) + "_" + to_string(g));
			}
		}
		next_iteration = false;
	}
	//system("pause");
}
void visualization6(int scene_num, PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<vector<Corres>> Instances, float resolution)
{
	//visulization
	pcl::visualization::PCLVisualizer viewer("GTM_CV");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src0(cloud_src, 0, 166, 237);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src1(cloud_src, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tar);
	//	  pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 128, 128, 128);
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 181, 181, 181);
	//    pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 139, 136, 120);

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y -= 100 * resolution;
	}
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src0, "cloud_src");
	viewer.addPointCloud(cloud_tar, rgb, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");

	//for (int i = 0; i < Mats.size(); i++)
	//{
	//	double line_R = (rand() % 255);
	//	double line_G = (rand() % 255);
	//	double line_B = (rand() % 255);
	//	double max_channel = std::max(line_R, std::max(line_G, line_B));
	//	line_R /= max_channel;
	//	line_G /= max_channel;
	//	line_B /= max_channel;
	//	for (size_t j = 0; j < Instances[i].size(); ++j)
	//	{
	//		int idx1 = Instances[i][j].source_idx;
	//		int idx2 = Instances[i][j].target_idx;
	//		//        cout << "m = " << j << endl;
	//		//        cout << "match1.source_idx = " << idx1 << "  match1.target_idx = " << idx2 << endl;
	//		//        cout << "match2.source_idx = " << idx3 << "  match2.target_idx = " << idx4 << endl;
	//		//        cout << "match3.source_idx = " << idx5 << "  match3.target_idx = " << idx6 << endl;
	//		std::stringstream SS_line_b1;
	//		SS_line_b1 << "line" << i << "_" << j;
	//		viewer.addLine< pcl::PointXYZ, pcl::PointXYZRGB>(cloud_src->points[idx1], cloud_tar->points[idx2], line_R, line_G, line_B, SS_line_b1.str());
	//	}
	//}

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y += 100 * resolution;
	}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);


	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src, 0, 255, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				//viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src0, "cloud_trans_src" + to_string(g));
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 0, 255, 0, "AABB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "AABB" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 255, 0, 0, "major eigen vector" + to_string(g));
				//viewer.addLine(center, y_axis, 255, 0, 0, "middle eigen vector" + to_string(g));
				//viewer.addLine(center, z_axis, 255, 0, 0, "minor eigen vector" + to_string(g));

			}
			for (int g = 0; g < Mats.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src2, Mats[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src2, 255, 0, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src2);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				viewer.addPointCloud(cloud_trans_src2, cloud_color_handler_src0, "cloud_trans_source2" + to_string(scene_num) + "_" + to_string(g));
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_source2" + to_string(scene_num) + "_" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 255, 0, 0, "AABB2" + to_string(scene_num) + "_" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB2" + to_string(scene_num) + "_" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "AABB2" + to_string(scene_num) + "_" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
				//viewer.addLine(center, x_axis, 0, 166, 237, "major eigen vector2" + to_string(scene_num) + "_" + to_string(g));
				//viewer.addLine(center, y_axis, 0, 166, 237, "middle eigen vector2" + to_string(scene_num) + "_" + to_string(g));
				//viewer.addLine(center, z_axis, 0, 166, 237, "minor eigen vector2" + to_string(scene_num) + "_" + to_string(g));
			}
		}
		next_iteration = false;
	}
	//system("pause");
}

void visualization7(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar_vis, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<Corres>& Corres, float resolution)
{
	//visulization
	pcl::visualization::PCLVisualizer viewer("GTM_CV");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 1, 165, 175);
	//pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_src, 0,0,255);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src1(cloud_src, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tar_vis);

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y -= 100 * resolution;
	}
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	//viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	viewer.addPointCloud(cloud_tar_vis, rgb, "cloud_tar_vis");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar_vis");


	double line_R = 0;
	double line_G = 0;
	double line_B = 0;
	double max_channel = 255;
	for (size_t i = 0; i < Corres.size(); ++i)
	{
		if (Corres[i].inlier)
		{
			line_G = 255;
		}
		else
		{
			line_R = 255;
		}
		line_R /= max_channel;
		line_G /= max_channel;
		line_B /= max_channel;
		int idx1 = Corres[i].source_idx;
		int idx2 = Corres[i].target_idx;
		std::stringstream SS_line_b1;
		SS_line_b1 << "line" << i;
		viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(cloud_src->points[idx1], cloud_tar->points[idx2], line_R, line_G, line_B, SS_line_b1.str());
		if (Corres[i].inlier) viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, SS_line_b1.str());//不透明度
		else viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, SS_line_b1.str());//不透明度
		//viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, SS_line_b1.str());//点大小
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, SS_line_b1.str());//线宽
	}

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y += 100 * resolution;
	}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);

	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src, 0, 255, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				//viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src" + to_string(g));
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 0.0, 1.0, 0.0, "AABB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "AABB" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
			}

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src2(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src2, Mats[Mats.size() - 1]);
			pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src2, 0, 166, 237);//255, 0, 0

			pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
			feature_extractor.setInputCloud(cloud_trans_src2);//设置输入点云
			feature_extractor.compute();//开始特征计算
			std::vector <float> moment_of_inertia;//存放惯性距的特征向量
			std::vector <float> eccentricity;//存放偏心率的特征向量
			pcl::PointXYZ min_point_AABB;
			pcl::PointXYZ max_point_AABB;
			pcl::PointXYZ min_point_OBB;
			pcl::PointXYZ max_point_OBB;
			pcl::PointXYZ position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			float major_value, middle_value, minor_value;
			Eigen::Vector3f major_vector, middle_vector, minor_vector;
			Eigen::Vector3f mass_center;
			feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
			feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
			feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
			feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
			feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
			feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
			feature_extractor.getMassCenter(mass_center);//计算质心
			viewer.initCameraParameters();
			//----------------------------------------------------------------------------------------------//
			viewer.addPointCloud(cloud_trans_src2, cloud_color_handler_src, "cloud_trans_source2_" + to_string(Mats.size() - 1));
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_source2_" + to_string(Mats.size() - 1));
			//----------------------------------------------------------------------------------------------//
			viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB2_" + to_string(Mats.size() - 1));
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB2_" + to_string(Mats.size() - 1));//不透明度
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "AABB2_" + to_string(Mats.size() - 1));//线宽
			viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
			pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
			pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
			pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
			pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
		}
		next_iteration = false;
	}
	//system("pause");
}

void visualization8(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar_vis, PointCloudPtr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<Corres>& match1, vector<Corres>& match2, vector<Corres>& match3, float resolution)
{
	//visulization
	pcl::visualization::PCLVisualizer viewer("GTM_CV");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 0, 166, 237);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tar_vis);

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y -= 100 * resolution;
	}
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar_vis, rgb, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");
	for (size_t j = 0; j < 1; j++)
	{
		int idx1 = match1[j].source_idx;
		int idx2 = match1[j].target_idx;
		int idx3 = match2[j].source_idx;
		int idx4 = match2[j].target_idx;
		int idx5 = match3[j].source_idx;
		int idx6 = match3[j].target_idx;
		std::stringstream SS_line_b1;
		SS_line_b1 << "line1" << j;
		std::stringstream SS_line_b2;
		SS_line_b2 << "line2" << j;
		std::stringstream SS_line_b3;
		SS_line_b3 << "line3" << j;
		double line_R = (rand() % 255);
		double line_G = (rand() % 255);
		double line_B = (rand() % 255);
		double max_channel = std::max(line_R, std::max(line_G, line_B));
		line_R /= max_channel;
		line_G /= max_channel;
		line_B /= max_channel;
		//double line_R = 79, line_G = 79, line_B = 79;
		viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(cloud_src->points[idx1], cloud_tar->points[idx2], line_R, line_G, line_B, SS_line_b1.str());
		viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(cloud_src->points[idx3], cloud_tar->points[idx4], line_R, line_G, line_B, SS_line_b2.str());
		viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(cloud_src->points[idx5], cloud_tar->points[idx6], line_R, line_G, line_B, SS_line_b3.str());

	}
	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y += 100 * resolution;
	}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);

	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src, 0, 255, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				//viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src" + to_string(g));
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 0.0, 1.0, 0.0, "AABB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "AABB" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
			}

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src2(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src2, Mats[Mats.size() - 1]);
			pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src2, 255, 0, 0);

			pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
			feature_extractor.setInputCloud(cloud_trans_src2);//设置输入点云
			feature_extractor.compute();//开始特征计算
			std::vector <float> moment_of_inertia;//存放惯性距的特征向量
			std::vector <float> eccentricity;//存放偏心率的特征向量
			pcl::PointXYZ min_point_AABB;
			pcl::PointXYZ max_point_AABB;
			pcl::PointXYZ min_point_OBB;
			pcl::PointXYZ max_point_OBB;
			pcl::PointXYZ position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			float major_value, middle_value, minor_value;
			Eigen::Vector3f major_vector, middle_vector, minor_vector;
			Eigen::Vector3f mass_center;
			feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
			feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
			feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
			feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
			feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
			feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
			feature_extractor.getMassCenter(mass_center);//计算质心
			viewer.initCameraParameters();
			//----------------------------------------------------------------------------------------------//
			viewer.addPointCloud(cloud_trans_src2, cloud_color_handler_src, "cloud_trans_source2_" + to_string(Mats.size() - 1));
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_source2_" + to_string(Mats.size() - 1));
			//----------------------------------------------------------------------------------------------//
			viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB2_" + to_string(Mats.size() - 1));
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB2_" + to_string(Mats.size() - 1));//不透明度
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "AABB2_" + to_string(Mats.size() - 1));//线宽
			viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
			pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
			pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
			pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
			pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
		}
		next_iteration = false;
	}
	//system("pause");
}
void visualization9(PointCloudPtr cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tar, vector<Eigen::Matrix4f>& Mats, vector<Eigen::Matrix4f>& GTs, vector<Corres>& Corres, float resolution)
{
	//visulization
	pcl::visualization::PCLVisualizer viewer("GTM_CV");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 0, 166, 237);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src1(cloud_src, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tar);

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y -= 100 * resolution;
	}
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, rgb, "cloud_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tar");

	double line_R = 0;
	double line_G = 255;
	double line_B = 0;
	double max_channel = std::max(line_R, std::max(line_G, line_B));
	line_R /= max_channel;
	line_G /= max_channel;
	line_B /= max_channel;
	for (size_t j = 0; j < Corres.size(); ++j)
	{
		int idx1 = Corres[j].source_idx;
		int idx2 = Corres[j].target_idx;
		std::stringstream SS_line_b1;
		SS_line_b1 << "line" << Mats.size() - 1 << "_" << j;
		viewer.addLine<pcl::PointXYZ, pcl::PointXYZRGB>(cloud_src->points[idx1], cloud_tar->points[idx2], line_R, line_G, line_B, SS_line_b1.str());
	}

	for (int i = 0; i < cloud_src->points.size(); i++) {
		cloud_src->points[i].y += 100 * resolution;
	}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);

	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			for (int g = 0; g < GTs.size(); g++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::transformPointCloud(*cloud_src, *cloud_trans_src, GTs[g]);
				pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src, 0, 255, 0);

				pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
				feature_extractor.setInputCloud(cloud_trans_src);//设置输入点云
				feature_extractor.compute();//开始特征计算
				std::vector <float> moment_of_inertia;//存放惯性距的特征向量
				std::vector <float> eccentricity;//存放偏心率的特征向量
				pcl::PointXYZ min_point_AABB;
				pcl::PointXYZ max_point_AABB;
				pcl::PointXYZ min_point_OBB;
				pcl::PointXYZ max_point_OBB;
				pcl::PointXYZ position_OBB;
				Eigen::Matrix3f rotational_matrix_OBB;
				float major_value, middle_value, minor_value;
				Eigen::Vector3f major_vector, middle_vector, minor_vector;
				Eigen::Vector3f mass_center;
				feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
				feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
				feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
				feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
				feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
				feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
				feature_extractor.getMassCenter(mass_center);//计算质心
				viewer.initCameraParameters();
				//----------------------------------------------------------------------------------------------//
				viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src" + to_string(g));
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_src" + to_string(g));
				//----------------------------------------------------------------------------------------------//
				viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 0.0, 1.0, 0.0, "AABB" + to_string(g));
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB" + to_string(g));//不透明度
				viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "AABB" + to_string(g));//线宽
				viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
				pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
				pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
				pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
				pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
			}

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src2(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src2, Mats[Mats.size() - 1]);
			pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_trans_src2, 255, 0, 0);

			pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;//实例化一个对象
			feature_extractor.setInputCloud(cloud_trans_src2);//设置输入点云
			feature_extractor.compute();//开始特征计算
			std::vector <float> moment_of_inertia;//存放惯性距的特征向量
			std::vector <float> eccentricity;//存放偏心率的特征向量
			pcl::PointXYZ min_point_AABB;
			pcl::PointXYZ max_point_AABB;
			pcl::PointXYZ min_point_OBB;
			pcl::PointXYZ max_point_OBB;
			pcl::PointXYZ position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			float major_value, middle_value, minor_value;
			Eigen::Vector3f major_vector, middle_vector, minor_vector;
			Eigen::Vector3f mass_center;
			feature_extractor.getMomentOfInertia(moment_of_inertia);//计算出的惯性矩
			feature_extractor.getEccentricity(eccentricity);//计算出的偏心率
			feature_extractor.getAABB(min_point_AABB, max_point_AABB);//计算轴对称边界盒子
			feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
			feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
			feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
			feature_extractor.getMassCenter(mass_center);//计算质心
			viewer.initCameraParameters();
			//----------------------------------------------------------------------------------------------//
			viewer.addPointCloud(cloud_trans_src2, cloud_color_handler_src1, "cloud_trans_source2_" + to_string(Mats.size() - 1));
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_trans_source2_" + to_string(Mats.size() - 1));
			//----------------------------------------------------------------------------------------------//
			viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB2_" + to_string(Mats.size() - 1));
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "AABB2_" + to_string(Mats.size() - 1));//不透明度
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "AABB2_" + to_string(Mats.size() - 1));//线宽
			viewer.setRepresentationToWireframeForAllActors();//将所有actor的可视化表示更改为线框表示
			pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
			pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
			pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
			pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
		}
		next_iteration = false;
	}
	//system("pause");
}