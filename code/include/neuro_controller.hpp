#ifndef NEURO_CAR_NEURO_CONTROLLER_HPP
#define NEURO_CAR_NEURO_CONTROLLER_HPP

#include <cstdint>

#include <car.hpp>

#include <controller.hpp>
#include <neural_network.hpp>

namespace NeuroCar {

class NeuroController : public Controller
{
    public:
        using NeuralNetwork = NeuroEvolution::NeuralNetwork;

    public:

        NeuroController();
        NeuroController(NeuralNetwork const & nn);
        virtual ~NeuroController();

        NeuralNetwork & getNeuralNetwork();
        NeuralNetwork const & getNeuralNetwork() const;
        void setNeuralNetwork(NeuralNetwork const & nn);
        void setDestination(b2Vec2 destination);

        virtual uint32_t updateFlags(Car * c) const override;

    private:
        NeuralNetwork m_neuralNetwork;
        b2Vec2 m_destination;
};

}

#endif //NEURO_CAR_NEURO_CONTROLLER_HPP
