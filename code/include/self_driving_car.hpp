#ifndef NEURO_CAR_SELF_DRIVING_CAR_HPP
#define NEURO_CAR_SELF_DRIVING_CAR_HPP

#include <car.hpp>
#include <dna.hpp>
#include <neuro_controller.hpp>

namespace NeuroCar {

//TODO
// - Create a world for each car from a common seed
// - Call the physic simulation with World::run
// -> how to get the feedback? Did the car reach its destination? Is it dead?
//      -> right now, the world only simulates and kills when the car hit an object
// -> For example, when the car reached its destination, stop the simulation or
//    expect the neural network to learn to stop in the target position?
//  - Car::setController
class SelfDrivingCar
{
    public:
        using NeuralNetwork = NeuroController::NeuralNetwork;

#if 0
        struct Params
        {
            double worldSeed;
            vec3 destination;
        };
#endif

    public:
        SelfDrivingCar();

        SelfDrivingCar(NeuralNetwork const & nn);

        NeuroController & getNeuroController();
        NeuroController const & getNeuroController() const;

        NeuralNetwork & getNeuralNetwork();
        NeuralNetwork const & getNeuralNetwork() const;

#if 0
        vec3 drive();
        vec3 const & getDestination() const;
        void setDestination(vec3 destination);
#endif

    private:
        Car * m_car; // std::shared_ptr<Car> ? (shared between generation)
        NeuroController m_neuroController;

#if 0
        vec3 m_destination;
#endif
};

class SelfDrivingCarDNA : public DNA<SelfDrivingCar, SelfDrivingCarDNA>
{
    public:
        SelfDrivingCarDNA(Subject subject = nullptr);

        virtual void randomize() override;
        virtual Fitness computeFitness() override;
        virtual Fitness getFitness() const override;
        virtual Subject crossover(SelfDrivingCarDNA const & partner) const override;
        virtual void mutate(MutationRate mutationRate) override;

    private:
        Fitness m_fitness;
};


}

#endif //NEURO_CAR_SELF_DRIVING_CAR_HPP
