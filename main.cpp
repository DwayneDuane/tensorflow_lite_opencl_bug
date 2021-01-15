#include "Model.hpp"
#include <fstream>

int main(int argc, char** argv)
{

    std::ifstream myFile; // creates stream myFile
    myFile.open("./input.txt"); // opens .txt file

    if (!myFile.is_open()) // check file is open, quit if not
    {
        std::cerr << "failed to open file\n";
        return 1;
    }

    std::vector<float> numberlist;
    float number = 0;
    while (myFile >> number)
        numberlist.push_back(number);

    std::cout << "total data size: " << numberlist.size() << std::endl;

    try {
	//To run with GPU delegate, change Model<CPUDelegate> to Model<GPUDelegate>
        Model<CPUDelegate> model("./dummy.tflite");
        model.Connect("Identity", "a");
        model.Connect("Identity_2", "a_1");

        constexpr int input_1_size = 100;

        auto DisplayResult = [&model](std::string const& name, int i)
        {
            auto output = model.GetOutput(name);

            std::cout << "Iter: " << i << " "<<name<<std::endl;
            for (auto v : output)
                std::cout << v << " ";
            std::cout <<std::endl<<"--------------------------"<<std::endl;
        };

        for (int i = 0; i < numberlist.size() / input_1_size; ++i) {
            std::vector<float> input_1(numberlist.begin() + i * input_1_size,
                numberlist.begin() + (i + 1) * input_1_size);
            model.FillInput("a_2", input_1);
            model.Forward();

            DisplayResult("Identity", i);
            DisplayResult("Identity_1", i);
            DisplayResult("Identity_2", i); 
            std::cout <<"**************************"<<std::endl;
        }
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return EXIT_SUCCESS;
}
