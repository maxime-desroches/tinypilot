#include <cassert>
#include <stdlib.h>
#include "common/util.h"
#include "snpemodel.h"

void PrintErrorStringAndExit() {
  const char* const errStr = zdl::DlSystem::getLastErrorString();
  std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
  std::exit(EXIT_FAILURE);
}

SNPEModel::SNPEModel(const char *path, float *loutput, size_t output_size, int runtime) {
  output = loutput;
#ifdef QCOM
  zdl::DlSystem::Runtime_t Runtime;
  if (runtime==USE_GPU_RUNTIME) {
    Runtime = zdl::DlSystem::Runtime_t::GPU;
  } else if (runtime==USE_DSP_RUNTIME) {
    Runtime = zdl::DlSystem::Runtime_t::DSP;
  } else {
    Runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  assert(zdl::SNPE::SNPEFactory::isRuntimeAvailable(Runtime));
#endif
  size_t model_size;
  model_data = (uint8_t *)read_file(path, &model_size);
  assert(model_data);

  // load model
  std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(model_data, model_size);
  if (!container) { PrintErrorStringAndExit(); }
  printf("loaded model with size: %lu\n", model_size);

  // create model runner
  zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
  while (!snpe) {
#ifdef QCOM
    snpe = snpeBuilder.setOutputLayers({})
                      .setRuntimeProcessor(Runtime)
                      .setUseUserSuppliedBuffers(true)
                      .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();
#else
    snpe = snpeBuilder.setOutputLayers({})
                      .setUseUserSuppliedBuffers(true)
                      .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();
#endif
    if (!snpe) std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
  }

  // get input and output names
  const auto &strListi_opt = snpe->getInputTensorNames();
  if (!strListi_opt) throw std::runtime_error("Error obtaining Input tensor names");
  const auto &strListi = *strListi_opt;
  //assert(strListi.size() == 1);
  const char *input_tensor_name = strListi.at(0);

  const auto &strListo_opt = snpe->getOutputTensorNames();
  if (!strListo_opt) throw std::runtime_error("Error obtaining Output tensor names");
  const auto &strListo = *strListo_opt;
  assert(strListo.size() == 1);
  const char *output_tensor_name = strListo.at(0);

  printf("model: %s -> %s\n", input_tensor_name, output_tensor_name);

  zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
  zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();

  // create input buffer
  {
    const auto &inputDims_opt = snpe->getInputDimensions(input_tensor_name);
    const zdl::DlSystem::TensorShape& bufferShape = *inputDims_opt;
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = sizeof(float);
    size_t product = 1;
    for (size_t i = 0; i < bufferShape.rank(); i++) product *= bufferShape[i];
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
      stride *= bufferShape[i];
      strides[i-1] = stride;
    }
    printf("input product is %lu\n", product);
    inputBuffer = ubFactory.createUserBuffer(NULL, product*sizeof(float), strides, &userBufferEncodingFloat);

    inputMap.add(input_tensor_name, inputBuffer.get());
  }

  // create output buffer
  {
    const zdl::DlSystem::TensorShape& bufferShape = snpe->getInputOutputBufferAttributes(output_tensor_name)->getDims();
    if (output_size != 0) {
      assert(output_size == bufferShape[1]);
    } else {
      output_size = bufferShape[1];
    }

    std::vector<size_t> outputStrides = {output_size * sizeof(float), sizeof(float)};
    outputBuffer = ubFactory.createUserBuffer(output, output_size * sizeof(float), outputStrides, &userBufferEncodingFloat);
    outputMap.add(output_tensor_name, outputBuffer.get());
  }
}

void SNPEModel::addRecurrent(float *state, int state_size) {
  recurrent = state;
  recurrentBuffer = this->addExtra(state, state_size, 3);
}

void SNPEModel::addTrafficConvention(float *state, int state_size) {
  trafficConvention = state;
  trafficConventionBuffer = this->addExtra(state, state_size, 2);
}

void SNPEModel::addDesire(float *state, int state_size) {
  desire = state;
  desireBuffer = this->addExtra(state, state_size, 1);
}

std::unique_ptr<zdl::DlSystem::IUserBuffer> SNPEModel::addExtra(float *state, int state_size, int idx) {
  // get input and output names
  const auto &strListi_opt = snpe->getInputTensorNames();
  if (!strListi_opt) throw std::runtime_error("Error obtaining Input tensor names");
  const auto &strListi = *strListi_opt;
  const char *input_tensor_name = strListi.at(idx);
  printf("adding index %d: %s\n", idx, input_tensor_name);

  zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
  zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  std::vector<size_t> retStrides = {state_size * sizeof(float), sizeof(float)};
  auto ret = ubFactory.createUserBuffer(state, state_size * sizeof(float), retStrides, &userBufferEncodingFloat);
  inputMap.add(input_tensor_name, ret.get());
  return ret;
}

void SNPEModel::execute(float *net_input_buf, int buf_size) {
#ifdef USE_THNEED
  if (thneed == NULL) {
    assert(inputBuffer->setBufferAddress(net_input_buf));
    thneed = new Thneed();
    if (!snpe->execute(inputMap, outputMap)) {
      PrintErrorStringAndExit();
    }
    t->stop();
  } else {
    float *inputs[4] = {state, trafficConvention, desire, input};
    t->execute(inputs, output);
  }

#else
  assert(inputBuffer->setBufferAddress(net_input_buf));
  if (!snpe->execute(inputMap, outputMap)) {
    PrintErrorStringAndExit();
  }
#endif
}

