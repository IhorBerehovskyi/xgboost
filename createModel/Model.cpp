#include "Model.hpp"
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <iostream>


Model::Model(const std::string& db_path) : db_path_(db_path), h_train_(nullptr), h_test_(nullptr), booster_(nullptr) {

}

Model::~Model() {
    
    XGBoosterFree(booster_);
    XGDMatrixFree(h_train_);
    XGDMatrixFree(h_test_);

}

// Load data from SQLite database
void Model::loadData() {

    sqlite3* db;
    sqlite3_stmt* stmt;

    if (sqlite3_open(db_path_.c_str(), &db) == SQLITE_OK) {
        std::string sql = "SELECT datetime, CO2Level FROM co2_levels";
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                double co2_level = sqlite3_column_double(stmt, 1);
                std::string datetime_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));

                // Convert date
                tm datetime = {};
                std::istringstream ss(datetime_str);
                ss >> std::get_time(&datetime, "%Y-%m-%d %H:%M:%S");

                std::vector<double> features = {
                    static_cast<double>(datetime.tm_hour),
                    static_cast<double>(datetime.tm_mday),
                    static_cast<double>(datetime.tm_mon + 1),
                    static_cast<double>(datetime.tm_wday)
                };

                X_.push_back(features);
                y_.push_back(co2_level);
            }
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }
}

// Create DMatrix
void Model::createDMatrices() {
    std::vector<float> X_flat;
    std::vector<float> y_flat(y_.begin(), y_.end());

    for (const auto& row : X_) {
        X_flat.insert(X_flat.end(), row.begin(), row.end());
    }

    int num_samples = X_.size();
    int num_features = X_[0].size();

    int train_size = static_cast<int>(num_samples * 0.8);
    int test_size = num_samples - train_size;

    XGDMatrixCreateFromMat(X_flat.data(), train_size, num_features, -1, &h_train_);
    XGDMatrixSetFloatInfo(h_train_, "label", y_flat.data(), train_size);

    XGDMatrixCreateFromMat(X_flat.data() + train_size * num_features, test_size, num_features, -1, &h_test_);
    XGDMatrixSetFloatInfo(h_test_, "label", y_flat.data() + train_size, test_size);
}

// Train the XGBoost model
void Model::trainModel(int num_rounds) {
    createDMatrices();

    XGBoosterCreate(&h_train_, 1, &booster_);
    XGBoosterSetParam(booster_, "objective", "reg:squarederror");
    XGBoosterSetParam(booster_, "eta", "0.1");
    XGBoosterSetParam(booster_, "max_depth", "5");
    XGBoosterSetParam(booster_, "verbosity", "1");

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_rounds; ++i) {
        XGBoosterUpdateOneIter(booster_, i, h_train_);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Training time: " << elapsed.count() << " seconds" << std::endl;
}

// Evaluate the model
void Model::evaluateModel() {
    bst_ulong out_len;
    const float* y_pred;
    XGBoosterPredict(booster_, h_test_, 0, 0, 0, &out_len, &y_pred);

    double mse = 0.0;
    int train_size = static_cast<int>(X_.size() * 0.8);

    for (bst_ulong i = 0; i < out_len; ++i) {
        mse += (y_[train_size + i] - y_pred[i]) * (y_[train_size + i] - y_pred[i]);
    }
    mse /= out_len;
    double rmse = std::sqrt(mse);

    double mean_y = std::accumulate(y_.begin(), y_.end(), 0.0) / y_.size();
    double std_dev_y = std::sqrt(std::accumulate(y_.begin(), y_.end(), 0.0, [mean_y](double sum, double val) {
        return sum + (val - mean_y) * (val - mean_y);
    }) / y_.size());

    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << rmse << std::endl;
    std::cout << "Standard Deviation of the target variable (y): " << std_dev_y << std::endl;
}

// Save the model
void Model::saveModel(const std::string& model_path) {
    XGBoosterSaveModel(booster_, model_path.c_str());
}
