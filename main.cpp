#include <iostream>
#include <format>
#include <vector>
#include <random>
#include <thread>
#include <stop_token>
#include <chrono>
#include <cmath>
#include "neural_network.h"

enum class CELL{EMPTY, AI, NON_AI};

int main()
{
    neural_network<1, 32, 32, 32, 1> sineNN(0.002);

    constexpr int rows = 80,
                  columns = 120;

    constexpr double minX = 0,
                     maxX = 3.14159 * 2,
                     minY = -1,
                     maxY = 1,
                     incrementPerColumn = (maxX - minX) / columns;

    CELL grid[rows][columns];

    auto resetGrid = [&grid]()
    {
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                grid[row][column] = CELL::EMPTY;
    };

    auto moveCursor = [](int row, int column)
    {
        std::cout << std::format("\033[{};{}H", row + 1, column + 1);
    };

    auto changeColor = [](int red, int green, int blue)
    {
        std::cout << std::format("\033[38;2;{};{};{}m", red, green, blue);
    };

    std::random_device rd;
    std::uniform_real_distribution<double> xDist(minX, maxX);

    std::cout << "\033[2J";
    std::cout << "\033[?25l";

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::jthread thr{[&](std::stop_token st)
    {
        while (!st.stop_requested())
        {
            resetGrid();

            for (int column = 0; column < columns; ++column)
            {
                const double x = minX + incrementPerColumn * column,
                             normalizedX = (x - minX) / (maxX - minX) * 2 - 1;

                const double predictedY = sineNN.predict(std::vector<double>(1, normalizedX))[0],
                             y = std::sin(x);

                const int predictedRow = rows - 1 - static_cast<int>((predictedY - minY) / (maxY - minY) * (rows - 1)),
                          actualRow = rows - 1 - static_cast<int>((y - minY) / (maxY - minY) * (rows - 1));

                grid[actualRow][column] = CELL::NON_AI;

                if ((predictedRow >= 0) && (predictedRow < rows))
                    grid[predictedRow][column] = CELL::AI;
            }

            for (int iter = 0; iter < 10; ++iter)
            {
                const double x = xDist(rd),
                             normalizedX = (x - minX) / (maxX - minX) * 2 - 1,
                             y = std::sin(x);
                sineNN.fit(std::vector<double>(1, normalizedX), std::vector<double>(1, y));
            }

            moveCursor(0, 0);
            for (int row = 0; row < rows; ++row)
            {
                for (int column = 0; column < columns; ++column)
                {
                    switch (grid[row][column])
                    {
                        case CELL::AI:
                            changeColor(0, 255, 255);
                            std::cout << 'o';
                            std::cout << "\033[0m";
                            break;
                        case CELL::NON_AI:
                            std::cout << 'o';
                            break;
                        case CELL::EMPTY:
                            std::cout << ' ';
                            break;
                    }
                }

                std::cout << '\n';
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }};
    std::cin.get();

    std::cout << "\033[?25h";
    std::cout << "\033[0m";
}