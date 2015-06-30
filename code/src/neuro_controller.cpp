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
    shape.push_back(11);
    shape.push_back(11);
    shape.push_back(4);
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

    // Adding raycast results as input
    for(auto d : c->getCollisionDists())
    {
        inputs.push_back(d);
    }

    // Adding angle to destination as input
    b2Vec2 carPos = c->getPos();
    double carAngle = c->getAngle() * 180.0 / M_PI;



    double deltax = m_destination.x - carPos.x;
    double deltay = m_destination.y - carPos.y;

    double angle = std::atan2(deltay, deltax) * 180.0 / M_PI;

    angle += carAngle + 90;

    angle = round(angle/90)*90;

    while(angle < -179)
    {
        angle += 360;
    }

    while(angle > 180)
    {
        angle -= 360;
    }

    //std::cout << angle << std::endl;

    inputs.push_back(angle);

    // Adding distance to destination as input
    /*double dist = std::sqrt(
        std::pow((m_destination.x - carPos.x), 2) +
        std::pow((m_destination.y - carPos.y), 2)
    );

    // FIXME: ugly
    //dist /= std::sqrt(100.0 * 100.0 + 80.0 * 80.0);

    inputs.push_back(dist);*/

    // Compute next decision
    Weights outputs = m_neuralNetwork.compute(inputs);

    uint32_t flags = 0;

    float threshold = 0.5;

    if(outputs[0] > threshold) flags |= Car::RIGHT;
    if(outputs[1] > threshold) flags |= Car::LEFT;
    if(outputs[2] > threshold) flags |= Car::FORWARD;
    if(outputs[3] > threshold) flags |= Car::BACKWARD;

    return flags;
}

}
