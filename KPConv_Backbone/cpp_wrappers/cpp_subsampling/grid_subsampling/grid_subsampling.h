

#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:

	// Elements
	// ********

	int count;
	PointXYZ point;
	vector<float> features;
	vector<unordered_map<int, int>> labels;
    // added by Du
    vector<PointXYZ> pts;
    vector<float> normals;
    vector<float> boundaries;
    vector<float> dirs;
    int feadim = 0;

	// Methods
	// *******

	// Constructor
	SampledData() 
	{ 
		count = 0; 
		point = PointXYZ();
	}

	SampledData(const size_t fdim, const size_t ldim)
	{
		count = 0;
		point = PointXYZ();
	    features = vector<float>(fdim);
	    labels = vector<unordered_map<int, int>>(ldim);
	    // Added by Du
	    normals = vector<float>();
	    dirs = vector<float>();
	    feadim = int(fdim);
	}

	// Method Update
	void update_all(const PointXYZ p, vector<float>::iterator f_begin, vector<int>::iterator l_begin)
	{
		count += 1;
		point += p;
		pts.push_back(p);
		transform(features.begin(), features.end(), f_begin, features.begin(), plus<float>());

        // Added by Du to extract normals and boundaries
        if (feadim > 3)
        {
            normals.insert(normals.end(), f_begin + 4, f_begin + 7);
            boundaries.push_back(*(f_begin + 7));
            dirs.insert(dirs.end(), f_begin + 8, f_begin + 11);
        }

		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_features(const PointXYZ p, vector<float>::iterator f_begin)
	{
		count += 1;
		point += p;
		pts.push_back(p);
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());

        // Added by Du to extract normals and boundaries
        if (feadim > 3)
        {
            normals.insert(normals.end(), f_begin + 4, f_begin + 7);
            boundaries.push_back(*(f_begin + 7));
            dirs.insert(dirs.end(), f_begin + 8, f_begin + 11);
        }

		return;
	}
	void update_classes(const PointXYZ p, vector<int>::iterator l_begin)
	{
		count += 1;
		point += p;
		pts.push_back(p);
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_points(const PointXYZ p)
	{
		count += 1;
		point += p;
		pts.push_back(p);
		return;
	}
};

void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose);

void batch_grid_subsampling(vector<PointXYZ>& original_points,
                            vector<PointXYZ>& subsampled_points,
                            vector<float>& original_features,
                            vector<float>& subsampled_features,
                            vector<int>& original_classes,
                            vector<int>& subsampled_classes,
                            vector<int>& original_batches,
                            vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p);


