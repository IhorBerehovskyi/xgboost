#include "Model.hpp"
#include <iostream>

int main() {
    std::string db_path = "co2_levels.db";
    Model model(db_path);

    model.loadData();
    model.trainModel(100);
    model.evaluateModel();
    model.saveModel("cpp_model.model");

    return 0;
}

