#include <neuro_controller.hpp>

namespace NeuroCar {

NeuroController::NeuroController():
    Controller(),
    m_neuralNetwork()
{

}

NeuroController::NeuroController(NeuralNetwork const & nn):
    Controller(),
    m_neuralNetwork(nn)
{

}

NeuroController::~NeuroController()
{

}

NeuroController::NeuralNetwork & NeuroController::getNeuralNetwork()
{
    return m_neuralNetwork;
}

NeuroController::NeuralNetwork const & NeuroController::getNeuralNetwork() const
{
    return m_neuralNetwork;
}

void NeuroController::setNeuralNetwork(NeuralNetwork const & nn)
{
    m_neuralNetwork = nn;
}

uint32_t NeuroController::updateFlags(Car * c) const
{
    using Weights = std::vector<NeuroEvolution::Weight>;

    Weights inputs;

    (void) c;

    // Create inputs from car state and sensors
    #if 0
    // Speed
    // Direction
    // Raycast
    #endif

    // Compute next decision
    Weights outputs = m_neuralNetwork.compute(inputs);

    uint32_t flags = 0;

    #if 0
    float threshold = 0.9;
    if(outputs[0] > threshold) flags |= Car::UP;
    if(outputs[1] > threshold) flags |= Car::DOWN;
    if(outputs[2] > threshold) flags |= Car::LEFT;
    if(outputs[3] > threshold) flags |= Car::RIGHT;
    #endif

    return flags;
}

}
