#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <random>
#include <time.h>
#include <map>

using namespace std;

struct Data
{
    vector<double> features;
    int label;
};

//Reads data from a csv file as data
void loadData(const string& filename, vector<Data> &data) 
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Unable to open file " << filename << endl;
        return;
    }

    string line;
    while(getline(file,line))
    {
        stringstream ss(line);
        string token;
        Data inputData;

        while(getline(ss, token, ','))
        {
            inputData.features.push_back(stod(token));
        }

        inputData.label = static_cast<int>(inputData.features.back());
        inputData.features.pop_back();

        data.push_back(inputData);
    }
}

//Code for the KNN Model
class KNNModel
{
    private:
        vector<Data> trainingData;
    public:
        void loadTrainingData(const vector<Data> &data)
        {
            trainingData = data;
        }

        int predict(const vector<double> inputFeatures, int k)
        {
            vector<pair<double, int>> distances;//pair<distance, label>

            //Calculate distances to all training points
            for (const auto& d : trainingData)
            {
                double distance = 0.0;
                for (int i = 0; i < d.features.size(); i++)
                    distance += pow(d.features[i] - inputFeatures[i], 2); 
                distance = sqrt(distance);

                distances.push_back({distance, d.label});
            }

            //Sort distances in ascending order
            sort(distances.begin(), distances.end(), [](const auto& a, const auto& b)
            {
                return a.first < b.first;
            });

            //Count the votes for each label among the k nearest neighbors
            map<int,int> labelVotes;
            for (int i = 0; i < k; i++)
                labelVotes[distances[i].second]++;

            //Find the label with the most votes
            int maxVotes = 0;
            int predictedLabel = 0;
            for (const auto& v : labelVotes)
            {
                if (v.second > maxVotes)
                {
                    maxVotes = v.second;
                    predictedLabel = v.first;
                }
            }

            return predictedLabel;
        }
};

int main()
{
    vector<Data> aggregateData;
    vector<Data> trainingData;
    vector<Data> testingData;

    loadData("hw6.data.csv", aggregateData);

    //Random distribution setup
    mt19937 rng(time(NULL));
    uniform_int_distribution<int> dist(0, 9);

    //Pushing data to testing or training randomly with a 30/70 split
    for(const auto d : aggregateData)
    {
        int roll = dist(rng);
        if (roll < 3)
            testingData.push_back(d);
        else
            trainingData.push_back(d);
    }
        
    KNNModel model;

    model.loadTrainingData(trainingData);

    vector<int> testingLabels;

    for (int i = 0; i < testingData.size(); i++)
    {
        testingLabels.push_back(model.predict(testingData[i].features, 1));
        cout << i << endl; //To check progress since this takes really long
    }

    cout << "Done computing labels" << endl;

    double correct = 0;
    for (int i = 0; i < testingLabels.size(); i++)
    {
        if (testingLabels[i] == testingData[i].label)
            correct++;
    }

    cout << "Accuracy: " << ((double)correct/testingLabels.size()) * 100 << "%" << endl;

    return 0;
}