#include <cstdint>

#include <omp.h>

#include <car.hpp>
#include <renderer.hpp>
#include <staticbox.hpp>
#include <world.hpp>

#include <evolution.hpp>

#include <memory>

//#include <evolving_string.hpp>
#include <neuro_controller.hpp>
#include <self_driving_car.hpp>

#include <iostream>
#include <io_utils.hpp>

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

    for(auto i = 0; i < 100; ++i)
    {
        // Create car
        std::shared_ptr<Car> car = std::make_shared<Car>(b2Vec2(10, 10), toRadian(carAngle), 2, 3, 18.0, angles);


        NeuroCar::Individual<NeuroCar::SelfDrivingCar> sdCar = NeuroCar::createIndividual<NeuroCar::SelfDrivingCar>();

        sdCar->setCar(car);
        sdCar->setDestination(destination);
        sdCar->setWorldSeed(seed);

        cars.push_back(sdCar);
    }

    NeuroCar::EvolutionParams params;
    params.mutationRate = 0.5;
    params.elitism = 1;

    NeuroCar::DNAs<NeuroCar::SelfDrivingCarDNA> dnas = NeuroCar::evolve<NeuroCar::SelfDrivingCarDNA>(cars, 1000, params);

    /*for(auto & dna: dnas)
    {
        auto subject = dna.getSubject();
        std::cout << subject->getGenes() << std::endl;
    }*/

}

int main(int argc, char const ** argv)
{
    int32_t nthreads = argc > 1 ? std::atoi(argv[1]) : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    carEvolution();
    return 0;
}
