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

void carEvolution(CarDef const & carDef, b2Vec2 const & destination, int32_t worldSeed);
void replayBest(CarDef const & carDef, b2Vec2 const & destination, int32_t worldSeed);

void carMain(int argc, char const **)
{
    static auto const toRadian = [](float32 degree)
    {
        return degree * M_PI / 180.0;
    };

    float32 carAngle = 0.0;

    // Create car
    CarDef carDef;
    carDef.initPos = b2Vec2(10, 10);
    carDef.initAngle = toRadian(carAngle);
    carDef.width = 2.0;
    carDef.height = 3.0;
    carDef.acceleration = 18.0;

    //angles to ray cast
    carDef.raycastAngles.push_back(0.0f);
    carDef.raycastAngles.push_back(b2_pi);

    carDef.raycastAngles.push_back(b2_pi/2.0f);
    carDef.raycastAngles.push_back(-b2_pi/2.0f);

    carDef.raycastAngles.push_back(b2_pi/4.0f);
    carDef.raycastAngles.push_back(-b2_pi/4.0f);

    carDef.raycastAngles.push_back(b2_pi/8.0f);
    carDef.raycastAngles.push_back(-b2_pi/8.0f);

    carDef.raycastAngles.push_back(3.0f*b2_pi/8.0f);
    carDef.raycastAngles.push_back(-3.0f*b2_pi/8.0f);

    b2Vec2 destination(100, 80);
    int32_t worldSeed = 0x42;

    if(argc == 2)
    {
        carEvolution(carDef, destination, worldSeed);
    }
    else
    {
        omp_set_num_threads(1);
        replayBest(carDef, destination, worldSeed);
    }
}


void carEvolution(CarDef const & carDef, b2Vec2 const & destination, int32_t worldSeed)
{
    NeuroCar::Population<NeuroCar::SelfDrivingCar> cars;

    for(auto i = 0; i < 100; ++i)
    {
        auto sdCar = NeuroCar::createIndividual<NeuroCar::SelfDrivingCar>();
        sdCar->setCar(std::make_shared<Car>(carDef));
        sdCar->setDestination(destination);
        sdCar->setWorldSeed(worldSeed);
        cars.push_back(sdCar);
    }

    NeuroCar::EvolutionParams<NeuroCar::SelfDrivingCarDNA> params;
    params.mutationRate = 0.5;
    params.elitism = 2;

    static auto const preGenHook = [](std::size_t i, NeuroCar::DNAs<NeuroCar::SelfDrivingCarDNA> const &)
    {
        std::cout << "Generation " << i << std::endl;
    };

    static auto const saveToFileHook = [](std::size_t i, NeuroCar::DNAs<NeuroCar::SelfDrivingCarDNA> const & dnas)
    {
        std::cout << "Saving generation " << i << "..." << std::endl;

        NeuroCar::SelfDrivingCarDNA const & bestDNA = *std::max_element(dnas.begin(), dnas.end(),
            [](NeuroCar::SelfDrivingCarDNA const & lhs, NeuroCar::SelfDrivingCarDNA const & rhs)
            {
                return lhs.getFitness() < rhs.getFitness();
            }
        );

        auto car = bestDNA.getSubject();
        std::cout << "Best DNA fitness: " << bestDNA.getFitness() << std::endl;
        NeuroCar::NeuroController const & nc = car->getNeuroController();
        NeuroCar::SelfDrivingCar::NeuralNetwork const & nn = nc.getNeuralNetwork();
        std::stringstream filename;
        filename << "best_nn_" << i << ".txt";
        NeuroEvolution::saveToFile(nn, filename.str());
        NeuroEvolution::saveToFile(nn, "last_best_nn.txt");
        //std::cout << nn << std::endl;
    };

    params.preGenHook  = preGenHook;
    params.postGenHook = saveToFileHook;

    auto dnas = NeuroCar::evolve<NeuroCar::SelfDrivingCarDNA>(cars, 1000, params);
}

void replayBest(CarDef const & carDef, b2Vec2 const & destination, int32_t worldSeed)
{
    NeuroEvolution::NeuralNetwork nn;

    if(!loadFromFile("last_best_nn.txt", nn))
    {
        std::cout << "Failed to reload nn from file " << std::endl;
        return;
    }

    std::cout << nn << std::endl;

    NeuroCar::Individual<NeuroCar::SelfDrivingCar> sdCar = NeuroCar::createIndividual<NeuroCar::SelfDrivingCar>();
    sdCar->setNeuroController(NeuroCar::NeuroController(nn));
    sdCar->setCar(std::make_shared<Car>(carDef));
    sdCar->setDestination(destination);
    sdCar->setWorldSeed(worldSeed);

    NeuroCar::SelfDrivingCarDNA dna(sdCar);
    for(auto i = 0u; i < 1; ++i)
    {
        auto fitness = dna.computeFitness();
        std::cout << "Fitness = " << fitness << std::endl;
        dna.reset();
    }
}

int main(int argc, char const ** argv)
{
    int32_t nthreads = argc > 1 ? std::atoi(argv[1]) : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    //NeuroCar::stringEvolution();
    carMain(argc, argv);
    return 0;
}
