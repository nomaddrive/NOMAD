#include "MapReader.hpp"
#include "json_serialization.hpp"
#include <cmath>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace {
madrona_gpudrive::Map *copyToArrayOnHostOrDevice(const madrona_gpudrive::Map *in,
                             madrona::ExecMode hostOrDevice) {
  madrona_gpudrive::Map *map = nullptr;

  if (hostOrDevice == madrona::ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
    map = static_cast<madrona_gpudrive::Map*>(madrona::cu::allocGPU(sizeof(madrona_gpudrive::Map)));
    cudaMemcpy(map, in, sizeof(madrona_gpudrive::Map), cudaMemcpyHostToDevice);
    auto error = cudaGetLastError();
    assert (error == cudaSuccess);
    
#else
    FATAL("Madrona was not compiled with CUDA support");
#endif
  } else {
    assert(hostOrDevice == madrona::ExecMode::CPU);

    // This is a copy from CPU to CPU and can be avoided by extracting and
    // releasing in's backing array. For the sake of symmetry with the CUDA
    // scenario, we nevertheless opt to copy the data.
    map = new madrona_gpudrive::Map();
    std::memcpy(map, in, sizeof(madrona_gpudrive::Map));
  }

  return map;
}
} // namespace

namespace madrona_gpudrive {

MapReader::MapReader(const std::string &pathToFile) : in_(pathToFile) {
  assert(in_.is_open());
  map_ = new madrona_gpudrive::Map();
}

MapReader::~MapReader() {
    delete map_;
}

void MapReader::doParse(float polylineReductionThreshold) {
  nlohmann::json rawJson;
  in_ >> rawJson;

  from_json(rawJson, *map_, polylineReductionThreshold);
}

static inline void mirrorXMap(madrona_gpudrive::Map *map) {
  // Mirror X: x -> -x; heading θ -> π - θ (wrapped); velocity x -> -x; means & goal positions mirrored.
  auto wrap = [](float h) {
    while (h > M_PI) h -= 2.f * M_PI;
    while (h < -M_PI) h += 2.f * M_PI;
    return h;
  };
  // Objects
  for (uint32_t i = 0; i < map->numObjects; i++) {
    auto &obj = map->objects[i];
    for (uint32_t k = 0; k < obj.numPositions; k++) {
      obj.position[k].x = -obj.position[k].x;
    }
    for (uint32_t k = 0; k < obj.numVelocities; k++) {
      obj.velocity[k].x = -obj.velocity[k].x;
    }
    for (uint32_t k = 0; k < obj.numHeadings; k++) {
      obj.heading[k] = wrap((float)M_PI - obj.heading[k]);
    }
    obj.goalPosition.x = -obj.goalPosition.x;
    obj.mean.x = -obj.mean.x;
  }
  // Roads
  for (uint32_t i = 0; i < map->numRoads; i++) {
    auto &road = map->roads[i];
    for (uint32_t k = 0; k < road.numPoints; k++) {
      road.geometry[k].x = -road.geometry[k].x;
    }
    // Only reverse the geometry points for non-lane types (e.g. RoadEdge, RoadLine).
    // RoadLane geometry defines the flow of traffic, which is preserved by the mirror
    // (forward remains forward). Reversing it would make the lane go backwards.
    if (road.type != EntityType::RoadLane) {
      std::reverse(road.geometry, road.geometry + road.numPoints);
    }
    road.mean.x = -road.mean.x;
  }
  map->mean.x = -map->mean.x;
}

madrona_gpudrive::Map* MapReader::parseAndWriteOut(const std::string &path,
              madrona::ExecMode executionMode, float polylineReductionThreshold, bool mirrorX) {
  MapReader reader(path);
  reader.doParse(polylineReductionThreshold);

  if (mirrorX) {
    mirrorXMap(reader.map_);
  }

  return copyToArrayOnHostOrDevice(reader.map_, executionMode);

} 
} // namespace madrona_gpudrive
