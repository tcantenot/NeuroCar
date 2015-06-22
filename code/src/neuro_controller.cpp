#include <neuro_controller.hpp>
#include <car.hpp>
#include <functions.hpp>

namespace NeuroCar {

NeuroController::NeuroController():
    Controller(),
    m_neuralNetwork()
{
    //Shape
    NeuralNetwork::Shape shape;
    shape.push_back(1);
    shape.push_back(4);
    shape.push_back(4);
    m_neuralNetwork.setShape(shape);
    //m_neuralNetwork.setActivationFunc(NeuroEvolution::sigmoid);
    //m_neuralNetwork.setActivationFuncPrime(NeuroEvolution::sigmoid_prime);
    m_neuralNetwork.synthetize();
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

    int posX = c->getPos().x;
    inputs.push_back(posX);

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

    float threshold = 0.5;
    if(outputs[0] > threshold) flags |= Car::LEFT;
    if(outputs[1] > threshold) flags |= Car::RIGHT;
    if(outputs[2] > threshold) flags |= Car::FORWARD;
    if(outputs[3] > threshold) flags |= Car::BACKWARD;

    return flags;
}

}
