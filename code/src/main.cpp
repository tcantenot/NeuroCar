#include <cstdint>

#include <omp.h>

#include <car.hpp>
#include <renderer.hpp>
#include <staticbox.hpp>
#include <world.hpp>

#include <evolution.hpp>

#include <memory>

#include <evolving_string.hpp>
#include <neuro_controller.hpp>
#include <self_driving_car.hpp>

#include <iostream>
#include <sstream>
#include <io_utils.hpp>
#include <serialization.hpp>

void carEvolution()
{
    NeuroCar::Population<NeuroCar::SelfDrivingCar> cars;

    std::vector<float32> angles;
    //angles to ray cast
    angles.push_back(0.0f);
    angles.push_back(b2_pi);

    angles.push_back(b2_pi/2.0f);
    angles.push_back(-b2_pi/2.0f);

    angles.push_back(b2_pi/4.0f);
    angles.push_back(-b2_pi/4.0f);

    angles.push_back(b2_pi/8.0f);
    angles.push_back(-b2_pi/8.0f);

    angles.push_back(3.0f*b2_pi/8.0f);
    angles.push_back(-3.0f*b2_pi/8.0f);

    static auto const toRadian = [](float32 degree)
    {
        return degree * M_PI / 180.0;
    };

    float32 carAngle = 0.0;

    b2Vec2 destination(50, 40);
    uint32_t seed = 1;

    for(auto i = 0; i < 5; ++i)
    {
        // Create car
        std::shared_ptr<Car> car = std::make_shared<Car>(b2Vec2(10, 10), toRadian(carAngle), 2, 3, 18.0, angles);


        NeuroCar::Individual<NeuroCar::SelfDrivingCar> sdCar = NeuroCar::createIndividual<NeuroCar::SelfDrivingCar>();

        sdCar->setCar(car);
        sdCar->setDestination(destination);
        sdCar->setWorldSeed(seed);

        cars.push_back(sdCar);
    }

    NeuroCar::EvolutionParams<NeuroCar::SelfDrivingCarDNA> params;
    params.mutationRate = 0.5;
    params.elitism = 2;

    static auto const printHook = [](std::size_t i, NeuroCar::DNAs<NeuroCar::SelfDrivingCarDNA> const & dnas)
    {
        std::cout << "Hook: " << i << std::endl;

        NeuroCar::SelfDrivingCarDNA const & bestDNA = *std::max_element(dnas.begin(), dnas.end(),
            [](NeuroCar::SelfDrivingCarDNA const & lhs, NeuroCar::SelfDrivingCarDNA const & rhs)
            {
                return lhs.getFitness() < rhs.getFitness();
            }
        );

        auto car = bestDNA.getSubject();
        NeuroCar::NeuroController const & nc = car->getNeuroController();
        NeuroCar::SelfDrivingCar::NeuralNetwork const & nn = nc.getNeuralNetwork();
        std::stringstream filename;
        filename << "best_nn_" << i << ".txt";
        NeuroEvolution::saveToFile(nn, filename.str());
        NeuroEvolution::saveToFile(nn, "last_best_nn.txt");
        //std::cout << nn << std::endl;
    };

    params.postGenHook = printHook;

    NeuroCar::DNAs<NeuroCar::SelfDrivingCarDNA> dnas = NeuroCar::evolve<NeuroCar::SelfDrivingCarDNA>(cars, 1000, params);
}

void replayBest()
{
    NeuroCar::Population<NeuroCar::SelfDrivingCar> cars;

    std::vector<float32> angles;
    //angles to ray cast
    angles.push_back(0.0f);
    angles.push_back(b2_pi);

    angles.push_back(b2_pi/2.0f);
    angles.push_back(-b2_pi/2.0f);

    angles.push_back(b2_pi/4.0f);
    angles.push_back(-b2_pi/4.0f);

    angles.push_back(b2_pi/8.0f);
    angles.push_back(-b2_pi/8.0f);

    angles.push_back(3.0f*b2_pi/8.0f);
    angles.push_back(-3.0f*b2_pi/8.0f);

    static auto const toRadian = [](float32 degree)
    {
        return degree * M_PI / 180.0;
    };

    float32 carAngle = 0.0;

    b2Vec2 destination(50, 40);
    uint32_t seed = 1;

    // Create car
    std::shared_ptr<Car> car = std::make_shared<Car>(b2Vec2(10, 10), toRadian(carAngle), 2, 3, 18.0, angles);

#if 1
    NeuroCar::Individual<NeuroCar::SelfDrivingCar> sdCar = NeuroCar::createIndividual<NeuroCar::SelfDrivingCar>();

    NeuroEvolution::NeuralNetwork nn;

    if(!loadFromFile("last_best_nn.txt", nn))
    {
        std::cout << "Failed to reload nn from file " << std::endl;
    }

    sdCar->setNeuroController(NeuroCar::NeuroController(nn));
    sdCar->setCar(car);
    sdCar->setDestination(destination);
    sdCar->setWorldSeed(seed);

    NeuroCar::SelfDrivingCarDNA dna(sdCar);
    for(auto i = 0u; i < 2; ++i)
    {
        auto sdcar = dna.getSubject();
        Car * car = sdcar->getCar().get();
        //NeuroCar::NeuroController const & nc = car->getNeuroController();
        //NeuroCar::SelfDrivingCar::NeuralNetwork const & nn = nc.getNeuralNetwork();
        //std::cout << nn << std::endl;
        //std::cout << "Car: " <<  car << std::endl;
        //std::cout << *car << std::endl;

        dna.computeFitness();
        dna.reset();
        std::cout << std::endl << std::endl;
    }
#endif
}

int main(int argc, char const ** argv)
{
    int32_t nthreads = argc > 1 ? std::atoi(argv[1]) : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    //NeuroCar::stringEvolution();
    if(argc == 2)
    {
        carEvolution();
    }
    else
    {
        omp_set_num_threads(1);
        std::cout << "Replay best" << std::endl;
        std::cout << std::endl;
        replayBest();
    }

    return 0;
}
