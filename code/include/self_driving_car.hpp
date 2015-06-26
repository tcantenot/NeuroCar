#ifndef NEURO_CAR_SELF_DRIVING_CAR_HPP
#define NEURO_CAR_SELF_DRIVING_CAR_HPP

#include <memory>

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

    public:
        SelfDrivingCar();

        NeuroController & getNeuroController();
        NeuroController const & getNeuroController() const;

        void setNeuroController(NeuroController nc);

        NeuralNetwork & getNeuralNetwork();
        NeuralNetwork const & getNeuralNetwork() const;

        b2Vec2 const & getDestination() const;
        void setDestination(b2Vec2 destination);

        void setWorldSeed(uint32_t seed);
        uint32_t getWorldSeed();

        std::shared_ptr<Car> & getCar();
        std::shared_ptr<Car> const & getCar() const ;

        void setCar(std::shared_ptr<Car> car);


    private:
        std::shared_ptr<Car> m_car;
        NeuroController m_neuroController;

        uint32_t m_worldSeed;
        b2Vec2 m_destination;
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
