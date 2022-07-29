#include "dataset_reader.h"
#include "dior_dataset_reader.h"
#include "euroc_dataset_reader.h"
#include "sensors_dataset_reader.h"

std::unique_ptr<DatasetReader>
DatasetReader::create_reader(const std::string &type, const std::string &filename) {
    // if (type == "sensors") {
    //     return std::make_unique<SensorsDatasetReader>(filename);
    // } else if (type == "dior") {
    //     return std::make_unique<SensorsDatasetReader>(filename);
    // } else
    return nullptr;
}