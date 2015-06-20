#ifndef NEURO_CAR_NEURO_CONTROLLER_HPP
#define NEURO_CAR_NEURO_CONTROLLER_HPP

#include <cstdint>

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

        virtual uint32_t updateFlags(Car * c) const override;

    private:
        NeuralNetwork m_neuralNetwork;
};

}

#endif //NEURO_CAR_NEURO_CONTROLLER_HPP
