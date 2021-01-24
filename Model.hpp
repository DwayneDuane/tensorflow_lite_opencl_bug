#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

class Model {
private:
    enum class Polarity {
        In,
        Out
    };

    auto FindIOByName(std::string name, Polarity polarity) const
    {
        std::vector<TFLiteIODetail> const* ptr = &mInputMasterList;
        if (polarity == Polarity::Out)
            ptr = &mOutputMasterList;

        auto it = std::find_if(ptr->cbegin(), ptr->cend(), [&name](TFLiteIODetail const& detail) { return detail.name == name; });
        if (it == ptr->cend())
            throw std::runtime_error("Cannot find IO: " + name);
        return std::make_pair(ptr->cbegin(), it);
    }

    auto FindIOElemCountByName(std::string name, Polarity polarity) const
    {
        auto itPair = FindIOByName(name, polarity);
        return itPair.second->NumElements();
    }

    auto FindIOIndexByName(std::string name, Polarity polarity) const
    {
        auto itPair = FindIOByName(name, polarity);
        return std::distance(itPair.first, itPair.second);
    }

public:
    Model(std::string tfliteModelFileName)
        : mOptions { TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDeleter {} }
        , mModel { TfLiteModelCreateFromFile(tfliteModelFileName.c_str()), TfLiteModelDeleter {} }
        , mInterpreter { nullptr, TfLiteInterpreterDeleter {} }
        , mXNNPackDelegate { nullptr, XNNPACKDelegateDeleter {} }
    {
        if (mModel == nullptr)
            throw std::runtime_error("Unable to load tflite model");

        //Enable XNNPack delegate
        TfLiteXNNPackDelegateOptions xnnpackOpts = TfLiteXNNPackDelegateOptionsDefault();
        //xnnpackOpts.num_threads = 1;
        mXNNPackDelegate.reset(TfLiteXNNPackDelegateCreate(&xnnpackOpts));
        if (mXNNPackDelegate == nullptr)
            throw std::runtime_error("Unable to initialized xnnpack delegate");

        TfLiteInterpreterOptionsAddDelegate(mOptions.get(), mXNNPackDelegate.get());

        //Call to TfLiteInterpreterOptionsAddDelegate goes here.
        mInterpreter.reset(TfLiteInterpreterCreate(mModel.get(), mOptions.get()));

        if (mInterpreter == nullptr)
            throw std::runtime_error("Unable to create interpreter");

        auto status = TfLiteInterpreterAllocateTensors(mInterpreter.get());
        if (status != kTfLiteOk)
            throw std::runtime_error("Unable to allocate tensors");

        auto numInputs = TfLiteInterpreterGetInputTensorCount(mInterpreter.get());
        for (int i = 0; i < numInputs; ++i) {
            mInputMasterList.emplace_back(TfLiteInterpreterGetInputTensor(mInterpreter.get(), i));
        }

        auto numOutputs = TfLiteInterpreterGetOutputTensorCount(mInterpreter.get());
        for (int i = 0; i < numOutputs; ++i) {
            mOutputMasterList.emplace_back(TfLiteInterpreterGetOutputTensor(mInterpreter.get(), i));
        }
    }

    void FillInput(std::string name, std::vector<float> const& data)
    {
        auto inputIndex = FindIOIndexByName(name, Polarity::In);
        TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(mInterpreter.get(), inputIndex);
        auto status = TfLiteTensorCopyFromBuffer(inputTensor, data.data(), data.size() * sizeof(float));
        if (status != kTfLiteOk)
            throw std::runtime_error("TfLiteTensorCopyFromBuffer failed");
    }

    void Connect(std::string rnnOut, std::string rnnIn)
    {
        {
            auto it = mRnnFeedbackMap.find(rnnOut);
            if (it != mRnnFeedbackMap.end())
                throw std::runtime_error("Output: " + rnnOut + " already exists in mRnnFeedbackMap");
        }

        {
            auto it = mStateMap.find(rnnIn);
            if (it != mStateMap.end())
                throw std::runtime_error("Intput: " + rnnIn + " already exists in mRnnFeedbackMap");
        }

        auto tensorElemCount = FindIOElemCountByName(rnnIn, Polarity::In);

        mRnnFeedbackMap[rnnOut] = rnnIn;
        mStateMap[rnnIn] = std::vector<float>(tensorElemCount, 0.f);
    }

    void Forward()
    {
        for (auto const& [name, data] : mStateMap) {
            FillInput(name, data);
        }

        auto start = std::chrono::high_resolution_clock::now();
        TfLiteInterpreterInvoke(mInterpreter.get());
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time - " << (end - start).count() * 1e-6f << "ms" << std::endl;

        for (auto const& [outName, inName] : mRnnFeedbackMap) {
            mStateMap.at(inName) = ExtractOutputFromTFLite(outName);
        }
    }

    std::vector<float> GetOutput(std::string name) const
    {
        auto it = mRnnFeedbackMap.find(name);
        if (it != mRnnFeedbackMap.end()) {
            return mStateMap.at(mRnnFeedbackMap.at(name));
        }
        return ExtractOutputFromTFLite(name);
    }

private:
    struct TfLiteInterpreterOptionsDeleter {
        void operator()(TfLiteInterpreterOptions* opts) const noexcept
        {
            if (opts)
                TfLiteInterpreterOptionsDelete(opts);
        }
    };

    struct TfLiteModelDeleter {
        void operator()(TfLiteModel* model) const noexcept
        {
            if (model)
                TfLiteModelDelete(model);
        }
    };

    struct TfLiteInterpreterDeleter {
        void operator()(TfLiteInterpreter* interpreter) const noexcept
        {
            if (interpreter)
                TfLiteInterpreterDelete(interpreter);
        }
    };

    struct XNNPACKDelegateDeleter {
        void operator()(TfLiteDelegate* delegate) const noexcept
        {
            if (delegate)
                TfLiteXNNPackDelegateDelete(delegate);
        }
    };

    std::unique_ptr<TfLiteInterpreterOptions, TfLiteInterpreterOptionsDeleter> mOptions;
    std::unique_ptr<TfLiteModel, TfLiteModelDeleter> mModel;
    std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> mInterpreter;
    std::unique_ptr<TfLiteDelegate, XNNPACKDelegateDeleter> mXNNPackDelegate;

    struct TFLiteIODetail {
        TFLiteIODetail(TfLiteTensor const* tensor)
            : name { TfLiteTensorName(tensor) }
            , dims(TfLiteTensorNumDims(tensor))
        {
            for (unsigned i = 0; i < dims.size(); ++i)
                dims[i] = TfLiteTensorDim(tensor, i);

            bool allPositive = std::all_of(dims.cbegin(), dims.cend(), [](int v) { return v > 0; });
            if (!allPositive)
                throw std::runtime_error("Cannot handle non positive dimensions");
        }

        int NumElements() const
        {
            return std::accumulate(dims.cbegin(), dims.cend(), 1, std::multiplies<int>());
        }

        std::string name;
        std::vector<int> dims;
    };

    mutable std::vector<TFLiteIODetail> mInputMasterList;
    mutable std::vector<TFLiteIODetail> mOutputMasterList;

    mutable std::map<std::string, std::vector<float>> mStateMap;
    mutable std::map<std::string, std::string> mRnnFeedbackMap;

    std::vector<float> ExtractOutputFromTFLite(std::string name) const
    {
        auto itPair = FindIOByName(name, Polarity::Out);
        auto tensorElemCount = itPair.second->NumElements();

        std::vector<float> output(tensorElemCount);

        auto outIndex = std::distance(itPair.first, itPair.second);
        TfLiteTensor const* outputTensor = TfLiteInterpreterGetOutputTensor(mInterpreter.get(), outIndex);

        auto status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size() * sizeof(float));
        if (status != kTfLiteOk)
            throw std::runtime_error("TfLiteTensorCopyToBuffer failed");

        return output;
    }
};
