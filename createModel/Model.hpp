#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>
#include <xgboost/c_api.h>
#include <sqlite3.h>
#include <chrono>

class Model {
public:
    Model(const std::string& db_path);
    ~Model();

    void loadData();
    void trainModel(int num_rounds = 100);
    void evaluateModel();
    void saveModel(const std::string& model_path);

private:
    std::string db_path_;
    std::vector<double> y_;
    std::vector<std::vector<double>> X_;
    DMatrixHandle h_train_;
    DMatrixHandle h_test_;
    BoosterHandle booster_;

    void createDMatrices();
};

#endif
