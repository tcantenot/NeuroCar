#include <neuro_controller.hpp>
#include <functions.hpp>

#include <cmath>

#include <iostream>
namespace NeuroCar {

NeuroController::NeuroController():
    Controller(),
    m_neuralNetwork()
{
    //Shape
    NeuralNetwork::Shape shape;
    shape.push_back(1);
    shape.push_back(2);
    shape.push_back(2);
    m_neuralNetwork.setShape(shape);
    m_neuralNetwork.setActivationFunc(NeuroEvolution::sigmoid);
    m_neuralNetwork.setActivationFuncPrime(NeuroEvolution::sigmoid_prime);
    m_neuralNetwork.setMinStartWeight(-1.0);
    m_neuralNetwork.setMaxStartWeight(1.0);
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

void NeuroController::setDestination(b2Vec2 destination)
{
    m_destination = destination;
}

uint32_t NeuroController::updateFlags(Car * c) const
{
    using Weights = std::vector<NeuroEvolution::Weight>;

    Weights inputs;

    std::vector<float32> dists = c->getDist();

    // Adding raycast results as input
    /*for (auto it = dists.begin(); it != dists.end();++it)
    {
        inputs.push_back(*it);
    }*/

    // Adding angle to destination as input
    b2Vec2 carPos = c->getPos();
    double carAngle = c->getAngle() * 180.0 / M_PI;

    while (carAngle < -180)
    {
        carAngle += 360;
    }

    while (carAngle > 180)
    {
        carAngle -= 360;
    }


    double deltax = m_destination.x - carPos.x;
    double deltay = m_destination.y - carPos.y;

    double angle = atan2(deltay, deltax) * 180 / M_PI;

    angle += carAngle;

    //inputs.push_back(angle);

    // Adding distance to destination as input
    double dist = sqrt(pow((m_destination.x - carPos.x), 2) + pow((m_destination.y - carPos.y), 2)) / sqrt(100*100 +80*80);
    //std::cout << dist << std::endl;
    inputs.push_back(dist);

    // Compute next decision
    Weights outputs = m_neuralNetwork.compute(inputs);

    uint32_t flags = 0;

    float threshold = 0.5;
    //std::cout << outputs[0] << std::endl;
    //std::cout << outputs[1] << std::endl;

    if(outputs[0] > threshold) flags |= Car::FORWARD;
    if(outputs[1] > threshold) flags |= Car::BACKWARD;
    //if(outputs[2] > threshold) flags |= Car::FORWARD;
    //if(outputs[3] > threshold) flags |= Car::BACKWARD;

    return flags;
}

}
