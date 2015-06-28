#ifndef CMD_OPTIONS_HPP
#define CMD_OPTIONS_HPP

#include <algorithm>
#include <sstream>


char * getCmdOption(char ** begin, char ** end, std::string const & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }

    return nullptr;
}

char * getCmdOption(int argc, char ** argv, std::string const & option)
{
    return getCmdOption(argv, argv + argc, option);
}

template <typename T>
bool getCmdOption(int argc, char ** argv, std::string const & option, T & value)
{
    if(char * val = getCmdOption(argv, argv + argc, option))
    {
        std::stringstream ss;
        ss << val;
        if(ss >> value)
        {
            return true;
        }
    }

    return false;
}


bool cmdOptionExists(char ** begin, char ** end, std::string const & option)
{
    return std::find(begin, end, option) != end;
}

template <typename ...Opts>
bool cmdOptionExists(char ** begin, char ** end, std::string const & option, Opts && ...opts)
{
    return cmdOptionExists(begin, end, option) ||
           cmdOptionExists(begin, end, std::forward<Opts>(opts)...);
}

bool cmdOptionExists(int argc, char ** argv, std::string const & option)
{
    return cmdOptionExists(argv, argv + argc, option);
}

template <typename ...Opts>
bool cmdOptionExists(int argc, char ** argv, std::string const & option, Opts && ...opts)
{
    return cmdOptionExists(argv, argv + argc, option, std::forward<Opts>(opts)...);
}


#endif //NEURO_CAR_CMD_OPTIONS_HPP
