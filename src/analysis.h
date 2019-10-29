#pragma once
#include <stdio.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <ctime>
#include <utility>

class Analysis {
private:
	static std::chrono::high_resolution_clock::time_point absoluteStart;
	static std::chrono::high_resolution_clock::time_point start;
	static std::chrono::high_resolution_clock::time_point finish;

	static std::vector<std::vector<int64_t>> durationList;
	static std::vector<std::pair<int, const char*>> labelList;
public:
	inline static void setAbsoluteStart() { absoluteStart = std::chrono::high_resolution_clock::now(); }
	inline static void begin() { start = std::chrono::high_resolution_clock::now(); }
	inline static void end(int index) { 
		finish = std::chrono::high_resolution_clock::now(); 

		if (index >= durationList.size()) {
			durationList.push_back(std::vector<int64_t>());
		}

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
		durationList[index].push_back(duration);
	}

	inline static void createLabel(int index, const char* label) {
		std::pair<int, const char*> tempLabel;
		tempLabel.first = index;
		tempLabel.second = label;

		labelList.push_back(tempLabel);
	}

	inline static void printAll(const int imageWidth, const int imageHeight) {
		time_t tempTime = time(NULL);

		printf("\n%s", ctime(&tempTime));
		printf("image resolution: %dx%d\n", imageWidth, imageHeight);

		for (int x = 0; x < durationList.size(); x++) {
			int64_t average = 0;

			for (int y = 0; y < durationList[x].size(); y++) {
				average += durationList[x][y];
			}

			for (int z = 0; z < labelList.size(); z++) {
				if (labelList[z].first == x) {
					printf("%s ", labelList[z].second);
				}
			}

			printf("[%d]: ", x);
			printf("%f\n", float(average / durationList[x].size()));
		}

		int64_t absoluteTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - absoluteStart).count();
		printf("Total: (%fs)\n", float(absoluteTime) / 1000000.0);
	}
};

std::chrono::high_resolution_clock::time_point Analysis::absoluteStart;
std::chrono::high_resolution_clock::time_point Analysis::start;
std::chrono::high_resolution_clock::time_point Analysis::finish;

std::vector<std::vector<int64_t>> Analysis::durationList;
std::vector<std::pair<int, const char*>> Analysis::labelList;