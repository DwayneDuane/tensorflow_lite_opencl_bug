#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

class CPUDelegate {
public:
    void Enable(tflite::Interpreter* interpreter)
    {
        auto status = interpreter->AllocateTensors();
        if (status != kTfLiteOk)
            throw std::runtime_error("interpreter->AllocateTensors() failed");
    }
};

class GPUDelegate {
private:
    struct TfLiteDelegateDeleter {
        void operator()(TfLiteDelegate* delegate) const noexcept
        {
            if (delegate)
                TfLiteGpuDelegateV2Delete(delegate);
        }
    };

    using delegate_ptr_type = std::unique_ptr<TfLiteDelegate, TfLiteDelegateDeleter>;

public:
    GPUDelegate()
    {
        TfLiteGpuDelegateOptionsV2 options;// = TfLiteGpuDelegateOptionsV2Default();
        options.is_precision_loss_allowed = 0;
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
	options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
        options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;

        options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;

        mDelegate = delegate_ptr_type(TfLiteGpuDelegateV2Create(&options), TfLiteDelegateDeleter {});
    }

    void Enable(tflite::Interpreter* interpreter)
    {
        auto status = interpreter->ModifyGraphWithDelegate(mDelegate.get());
        if (status != kTfLiteOk)
            throw std::runtime_error("interpreter->ModifyGraphWithDelegate failed");
    }

private:
    delegate_ptr_type mDelegate;
};

template <typename DelegateType>
class Model {
public:
    Model(std::string tfliteModelFileName)
        : mModel { tflite::FlatBufferModel::BuildFromFile(tfliteModelFileName.c_str()) }
    {
        if (mModel == nullptr)
            throw std::runtime_error("Unable to load tflite model");

        tflite::ops::builtin::BuiltinOpResolver op_resolver;
        tflite::InterpreterBuilder builder(*mModel, op_resolver);
        builder(&mInterpreter);

        if (mInterpreter == nullptr)
            throw std::runtime_error("Unable to create interpreter");

        mDelegate.Enable(mInterpreter.get());
    }

    void FillInput(std::string name, std::vector<float> const& data)
    {
        std::copy_n(data.data(), data.size(), GetRawInputTensorPointer(name, data.size()));
    }

    void Connect(std::string rnnOut, std::string rnnIn)
    {
        {
            auto it = mRnnFeedbackMap.find(rnnOut);
            if (it != mRnnFeedbackMap.end())
                throw std::runtime_error("Unable to insert into mRnnFeedbackMap");
        }

        {
            auto it = mStateMap.find(rnnIn);
            if (it != mStateMap.end())
                throw std::runtime_error("Unable to insert into mStateMap");
        }

        auto tensorElemCount = GetInputTensorSize(rnnIn);

        mRnnFeedbackMap[rnnOut] = rnnIn;
        mStateMap[rnnIn] = std::vector<float>(tensorElemCount, 0.f);
    }

    void Forward()
    {
        for (auto const& [name, data] : mStateMap) {
            FillInput(name, data);
        }

        auto start = std::chrono::high_resolution_clock::now();
        mInterpreter->Invoke();
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
    DelegateType mDelegate;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;

    mutable std::map<std::string, std::vector<float>> mStateMap;
    mutable std::map<std::string, std::string> mRnnFeedbackMap;

    std::vector<float> ExtractOutputFromTFLite(std::string name) const
    {
        auto tensorElemCount = GetOutputTensorSize(name);
        auto const* ptr = GetRawOutputTensorPointer(name, tensorElemCount);

        std::vector<float> output(tensorElemCount);
        std::copy_n(ptr, tensorElemCount, output.begin());
        return output;
    }

    float* GetRawInputTensorPointer(std::string name, size_t expectedElemCount) const
    {
        auto idx = FindInputTensorIndex(name);
        if (idx < 0)
            throw std::runtime_error("Input Tensor not found");

        auto* tensorPtr = mInterpreter->tensor(mInterpreter->inputs().at(idx));
        if (tensorPtr->type != kTfLiteFloat32)
            throw std::runtime_error("Model must have FP32 Input");

        const auto tensorElemCount = tflite::NumElements(tensorPtr);
        if (tensorElemCount != expectedElemCount)
            throw std::runtime_error("Incorrect Input number of elements");

        return mInterpreter->typed_input_tensor<float>(idx);
    }

    float* GetRawOutputTensorPointer(std::string name, size_t expectedElemCount) const
    {
        auto idx = FindOutputTensorIndex(name);
        if (idx < 0)
            throw std::runtime_error("Output Tensor not found");

        auto* tensorPtr = mInterpreter->tensor(mInterpreter->outputs().at(idx));
        if (tensorPtr->type != kTfLiteFloat32)
            throw std::runtime_error("Model must have FP32 Output");

        const auto tensorElemCount = tflite::NumElements(tensorPtr);
        if (tensorElemCount != expectedElemCount)
            throw std::runtime_error("Incorrect Output number of elements");

        return mInterpreter->typed_output_tensor<float>(idx);
    }

    auto GetInputTensorSize(std::string name) const
    {
        auto idx = FindInputTensorIndex(name);
        if (idx < 0)
            throw std::runtime_error("Input Tensor not found");
        auto* tensorPtr = mInterpreter->tensor(mInterpreter->inputs()[idx]);
        return tflite::NumElements(tensorPtr);
    }

    auto GetOutputTensorSize(std::string name) const
    {
        auto idx = FindOutputTensorIndex(name);
        if (idx < 0)
            throw std::runtime_error("Output Tensor not found");
        auto* tensorPtr = mInterpreter->tensor(mInterpreter->outputs()[idx]);
        return tflite::NumElements(tensorPtr);
    }

    int FindInputTensorIndex(std::string name) const
    {
        for (int k = 0; k < mInterpreter->inputs().size(); ++k) {
            if (std::string(mInterpreter->GetInputName(k)) == name)
                return k;
        }
        return -1;
    }

    int FindOutputTensorIndex(std::string name) const
    {
        for (int k = 0; k < mInterpreter->outputs().size(); ++k) {
            if (std::string(mInterpreter->GetOutputName(k)) == name)
                return k;
        }
        return -1;
    }
};
