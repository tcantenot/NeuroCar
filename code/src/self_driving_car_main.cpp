#include <self_driving_car_main.hpp>

#include <cstdint>
#include <iostream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <car.hpp>

#include <io_utils.hpp>
#include <serialization.hpp>

#include <evolution.hpp>
#include <evolving_string.hpp>
#include <neuro_controller.hpp>
#include <self_driving_car.hpp>

#include <cmd_options.hpp>


namespace NeuroCar {

namespace {

void carEvolution(
    CarDef const & carDef,
    DNAParams<SelfDrivingCarDNA> const & dnaParams,
    b2Vec2 const & destination,
    int32_t worldSeed,
    double mutationRate,
    uint32_t elitism,
    std::size_t nindividuals,
    std::size_t ngenerations,
    std::string const & filename
)
{
    Population<SelfDrivingCar> cars;

    for(auto i = 0u; i < nindividuals; ++i)
    {
        auto sdCar = createIndividual<SelfDrivingCar>();
        sdCar->setCar(std::make_shared<Car>(carDef));
        sdCar->setDestination(destination);
        sdCar->setWorldSeed(worldSeed);
        cars.push_back(sdCar);
    }

    static auto const preGenHook = [](
        std::size_t i, DNAs<SelfDrivingCarDNA> const &
    )
    {
        std::cout << "Generation " << i << std::endl;
    };

    auto const saveToFileHook = [&filename](
        std::size_t, DNAs<SelfDrivingCarDNA> const & dnas
    )
    {
        auto const & bestDNA = *std::max_element(dnas.begin(), dnas.end(),
            [](SelfDrivingCarDNA const & lhs, SelfDrivingCarDNA const & rhs)
            {
                return lhs.getFitness() < rhs.getFitness();
            }
        );

        std::cout << "Best DNA fitness: " << bestDNA.getFitness() << std::endl;

        auto car = bestDNA.getSubject();
        NeuroController const & nc = car->getNeuroController();
        SelfDrivingCar::NeuralNetwork const & nn = nc.getNeuralNetwork();

        //std::stringstream fname;
        //fname << "best_nn_" << i << ".txt";
        //NeuroEvolution::saveToFile(nn, fname.str());

        std::cout << "Saving to \"" << filename << "\"" << std::endl;
        NeuroEvolution::saveToFile(nn, filename);
    };

    EvolutionParams<SelfDrivingCarDNA> params;
    params.mutationRate = mutationRate;
    params.elitism      = elitism;
    params.preGenHook   = preGenHook;
    params.postGenHook  = saveToFileHook;
    params.dnaParams    = dnaParams;

    static auto const p = [](b2Vec2 const & v)
    {
        return std::string("(") + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
    };

    std::cout << "### NeuroCar Evolution ###" << std::endl;
    std::cout << "  Number of individuals: " << nindividuals                      << std::endl;
    std::cout << "  Number of generations: " << ngenerations                      << std::endl;
    std::cout << "  Mutation rate:         " << mutationRate                      << std::endl;
    std::cout << "  Elitism:               " << elitism                           << std::endl;
    std::cout << "  World seed:            " << worldSeed                         << std::endl;
    std::cout << "  World change interval: " << dnaParams.worldSeedChangeInterval << std::endl;
    std::cout << "  Starting point:        " << p(carDef.initPos)                 << std::endl;
    std::cout << "  Destination:           " << p(destination)                    << std::endl;
    std::cout << "  Output filename:       " << filename                          << std::endl;
    std::cout << std::endl;

    evolve<SelfDrivingCarDNA>(cars, ngenerations, params);
}

void replayBest(
    CarDef const & carDef,
    DNAParams<SelfDrivingCarDNA> const & dnaParams,
    b2Vec2 const & destination,
    int32_t worldSeed,
    std::string const & filename
)
{
    NeuroEvolution::NeuralNetwork nn;

    if(!loadFromFile(filename, nn))
    {
        std::cout << "Failed to reload nn from file \""
                  << filename << "\"" << std::endl;
        return;
    }

    std::cout << "### NeuroCar Replay ###" << std::endl;
    std::cout << nn << std::endl << std::endl;

    auto sdCar = createIndividual<SelfDrivingCar>();
    sdCar->setNeuroController(NeuroController(nn));
    sdCar->setCar(std::make_shared<Car>(carDef));
    sdCar->setDestination(destination);
    sdCar->setWorldSeed(worldSeed);

    SelfDrivingCarDNA dna(sdCar);
    dna.init(dnaParams);
    auto fitness = dna.computeFitness();
    std::cout << "Fitness = " << fitness << std::endl;
}

}

void selfDrivingCarMain(int argc, char ** argv)
{
    // "-h" option: Help
    if(cmdOptionExists(argc, argv, "-h"))
    {
        std::cout << "### NeuroCar Help ###" << std::endl;
        //TODO
        return;
    }

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

    int32_t worldSeed = 0;
    std::string filename = "last_best_nn.txt";
    double mutationRate = 0.01;
    uint32_t elitism = 2;
    std::size_t nindividuals = 100;
    std::size_t ngenerations = 100;

    DNAParams<SelfDrivingCarDNA> dnaParams;
    dnaParams.worldWidth  = 100;
    dnaParams.worldHeight = 80;
    dnaParams.worldNbObstacles = 15;
    dnaParams.worldSeedChangeInterval = 10;

    // "-c" option: World seed change interval
    int32_t ch = 0;
    if(getCmdOption(argc, argv, "-c", ch)) dnaParams.worldSeedChangeInterval = ch;

    // "-s" option: World seed
    int32_t seed = 0;
    if(getCmdOption(argc, argv, "-s", seed)) worldSeed = seed;

    // "-m" option: Mutation rate
    double mr = 0.0;
    if(getCmdOption(argc, argv, "-m", mr)) mutationRate = mr;

    // "-e" option: Mutation rate
    uint32_t e = 0.0;
    if(getCmdOption(argc, argv, "-e", e)) elitism = e;

    // "-i" option: Number of individuals
    std::size_t nindiv = 0;
    if(getCmdOption(argc, argv, "-i", nindiv)) nindividuals = nindiv;

    // "-g" option: Number of generations
    std::size_t ngen = 0;
    if(getCmdOption(argc, argv, "-g", ngen)) ngenerations = ngen;

    // "-r" or "--replay" option: replay best DNA
    if(cmdOptionExists(argc, argv, "-r", "--replay"))
    {
        char * f = getCmdOption(argc, argv, "-f");
        if(f) filename = f;

        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif

        replayBest(carDef, dnaParams, destination, worldSeed, filename);
    }
    else // Car evolution
    {
        int32_t nthreads = 1;

        // "--max-threads" option: use maximum number of threads?
        if(cmdOptionExists(argc, argv, "--max-threads"))
        {
            #ifdef _OPENMP
            nthreads = omp_get_max_threads();
            #endif
        }

        // "-t" option: number of threads to use
        int nt = 0;
        if(getCmdOption(argc, argv, "-t", nt)) nthreads = nt;

        #ifdef _OPENMP
        omp_set_num_threads(nthreads);
        #endif

        std::cout << "Running " << nthreads << " thread"
                  << (nthreads > 1 ? "s" : "") << std::endl;

        // Output file for the best DNA
        char * f = getCmdOption(argc, argv, "-f");
        if(f) filename = f;

        carEvolution(
            carDef,
            dnaParams,
            destination,
            worldSeed,
            mutationRate,
            elitism,
            nindividuals,
            ngenerations,
            filename
        );
    }
}

}
