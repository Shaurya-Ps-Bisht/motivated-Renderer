﻿#include "fastgltf/types.hpp"
#include "fastgltf/util.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "stb_image.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>
#include <vulkan/vulkan_core.h>

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine *engine,
                                                                      std::filesystem::path filepath)
{
    std::cout << "Loading Gltf: " << filepath << std::endl;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filepath);

    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser{};

    auto load = parser.loadBinaryGLTF(&data, filepath.parent_path(), gltfOptions);
    if (load)
    {
        gltf = std::move(load.get());
    }
    else
    {
        fmt::print("Failed to load GLTF: {} \n", fastgltf::to_underlying(load.error()));
        return {};
    }

    std::vector<std::shared_ptr<MeshAsset>> meshes;

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh : gltf.meshes)
    {
        MeshAsset newmesh;

        newmesh.name = mesh.name;

        indices.clear();
        vertices.clear();

        //
        for (auto &&p : mesh.primitives) // Universal references (when && used with auto or in Template Context)
        {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();
            // indices
            {
                fastgltf::Accessor &indexAccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexAccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(
                    gltf, indexAccessor, [&](std::uint32_t idx) { indices.push_back(idx + initial_vtx); });
            }

            // vertices POS
            {
                fastgltf::Accessor &posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
                    Vertex newvtx;
                    newvtx.position = v;
                    newvtx.normal = {1, 0, 0};
                    newvtx.color = glm::vec4{1.f};
                    newvtx.uv_x = 0;
                    newvtx.uv_y = 0;
                    vertices[initial_vtx + index] = newvtx;
                });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    gltf, gltf.accessors[(*normals).second],
                    [&](glm::vec3 v, size_t index) { vertices[initial_vtx + index].normal = v; });
            }

            // Uvs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                                                              [&](glm::vec2 v, size_t index) {
                                                                  vertices[initial_vtx + index].uv_x = v.x;
                                                                  vertices[initial_vtx + index].uv_y = v.y;
                                                              });
            }

            // colurs
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(
                    gltf, gltf.accessors[(*colors).second],
                    [&](glm::vec4 v, size_t index) { vertices[initial_vtx + index].color = v; });
            }

            newmesh.surfaces.push_back(newSurface);
        }

        constexpr bool OverrideColors = false;
        if (OverrideColors)
        {
            for (Vertex &vtx : vertices)
            {
                vtx.color = glm::vec4(vtx.normal, 1.f);
            }
        }

        newmesh.meshBuffers = engine->uploadMesh(indices, vertices);
        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newmesh)));
    }
    return meshes;
}

VkFilter extract_filter(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return VK_FILTER_NEAREST;

    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapLinear:
    case fastgltf::Filter::LinearMipMapNearest:
    default:
        return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine *engine, std::string_view filePath)
{
    fmt::print("Loading GLTF: {} \n", filePath);
    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF &file = *scene.get();

    fastgltf::Parser parser{};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble |
                                 fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto type = fastgltf::determineGltfFileType(&data);
    if (type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGLTF(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else if (type == fastgltf::GltfType::GLB)
    {
        auto load = parser.loadBinaryGLTF(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load glTF" << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else
    {
        std::cerr << "Failed to determine GLTF container" << std::endl;
        return {};
    }

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
                                                                     {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
                                                                     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};

    file.descriptorPool.init(engine->_device, gltf.materials.size(), sizes);

    for (fastgltf::Sampler &sampler : gltf.samplers)
    {
        VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = nullptr};
        sampl.maxLod = VK_LOD_CLAMP_NONE;
        sampl.minLod = 0;

        sampl.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        sampl.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        sampl.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        VkSampler newSampler;
        vkCreateSampler(engine->_device, &sampl, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }

    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;
    for (fastgltf::Image &image : gltf.images)
    {
        std::optional<AllocatedImage> img = load_image(engine, gltf, image);

        if (img.has_value())
        {
            images.push_back(*img);
            file.images[image.name.c_str()] = *img;
        }
        else
        {
            images.push_back(engine->_errorCheckerBoardImage);
            std::cout << "gltf failed to load texture" << image.name << std::endl;
        }
    }

    // for (fastgltf::Image &image : gltf.images)
    // {
    //     images.push_back(engine->_errorCheckerBoardImage);
    // }
    //
    file.materialDataBuffer =
        engine->create_buffer(sizeof(GLTFMetallic_roughness::MaterialConstants) * gltf.materials.size(),
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    int data_index = 0;
    GLTFMetallic_roughness::MaterialConstants *sceneMaterialConstants =
        (GLTFMetallic_roughness::MaterialConstants *)file.materialDataBuffer.info.pMappedData;

    for (fastgltf::Material &mat : gltf.materials)
    {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat;

        GLTFMetallic_roughness::MaterialConstants constants;
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metal_rough_factors.x = mat.pbrData.metallicFactor;
        constants.metal_rough_factors.y = mat.pbrData.roughnessFactor;
        sceneMaterialConstants[data_index] = constants;

        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend)
        {
            passType = MaterialPass::Transparent;
        }

        GLTFMetallic_roughness::MaterialResources materialResources;
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;

        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = data_index * sizeof(GLTFMetallic_roughness::MaterialConstants);

        if (mat.pbrData.baseColorTexture.has_value())
        {
            size_t img = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

            materialResources.colorImage = images[img];
            materialResources.colorSampler = file.samplers[sampler];
        }

        newMat->data = engine->metalRoughMaterial.write_material(engine->_device, passType, materialResources,
                                                                 file.descriptorPool);

        data_index++;
    }

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh : gltf.meshes)
    {
        std::shared_ptr<MeshAsset> newMesh = std::make_shared<MeshAsset>();
        meshes.push_back(newMesh);
        file.meshes[mesh.name.c_str()] = newMesh;
        newMesh->name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto &&p : mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();

            {
                fastgltf::Accessor &indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(
                    gltf, indexaccessor, [&](std::uint32_t idx) { indices.push_back(idx + initial_vtx); });
            }

            {
                fastgltf::Accessor &posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
                    Vertex newvtx;
                    newvtx.position = v;
                    newvtx.normal = {1, 0, 0};
                    newvtx.color = glm::vec4{1.f};
                    newvtx.uv_x = 0;
                    newvtx.uv_y = 0;
                    vertices[initial_vtx + index] = newvtx;
                });
            }

            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    gltf, gltf.accessors[(*normals).second],
                    [&](glm::vec3 v, size_t index) { vertices[initial_vtx + index].normal = v; });
            }

            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                                                              [&](glm::vec2 v, size_t index) {
                                                                  vertices[initial_vtx + index].uv_x = v.x;
                                                                  vertices[initial_vtx + index].uv_y = v.y;
                                                              });
            }

            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(
                    gltf, gltf.accessors[(*colors).second],
                    [&](glm::vec4 v, size_t index) { vertices[initial_vtx + index].color = v; });
            }
            if (p.materialIndex.has_value())
            {
                newSurface.material = materials[p.materialIndex.value()];
            }
            else
            {
                newSurface.material = materials[0];
            }

            glm::vec3 minPos = vertices[initial_vtx].position;
            glm::vec3 maxPos = vertices[initial_vtx].position;

            for (int i = initial_vtx; i < vertices.size(); i++)
            {
                minPos = glm::min(minPos, vertices[i].position);
                maxPos = glm::max(maxPos, vertices[i].position);
            }

            newSurface.bounds.origin = (maxPos + minPos) / 2.f;
            newSurface.bounds.extents = (maxPos - minPos) / 2.f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            newMesh->surfaces.push_back(newSurface);
        }

        newMesh->meshBuffers = engine->uploadMesh(indices, vertices);
    }

    // nodes
    for (fastgltf::Node &node : gltf.nodes)
    {
        std::shared_ptr<Node> newNode;

        if (node.meshIndex.has_value())
        {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode *>(newNode.get())->mesh = meshes[*node.meshIndex];
        }
        else
        {
            newNode = std::make_shared<Node>();
        }

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(fastgltf::visitor{[&](fastgltf::Node::TransformMatrix matrix) {
                                         memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
                                     },
                                     [&](fastgltf::Node::TRS transform) {
                                         glm::vec3 tl(transform.translation[0], transform.translation[1],
                                                      transform.translation[2]);
                                         glm::quat rot(transform.rotation[3], transform.rotation[0],
                                                       transform.rotation[1], transform.rotation[2]);
                                         glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                                         glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                                         glm::mat4 rm = glm::toMat4(rot);
                                         glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                                         newNode->localTransform = tm * rm * sm;
                                     }},
                   node.transform);
    }

    for (int i = 0; i < gltf.nodes.size(); i++)
    {
        fastgltf::Node &node = gltf.nodes[i];
        std::shared_ptr<Node> &sceneNode = nodes[i];

        for (auto &c : node.children)
        {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    for (auto &node : nodes)
    {
        if (node->parent.lock() == nullptr)
        {
            file.topNodes.push_back(node);
            node->refershTransform(glm::mat4{1.f});
        }
    }
    return scene;
}
void LoadedGLTF::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    for (auto &n : topNodes)
    {
        n->Draw(topMatrix, ctx);
    }
}
void LoadedGLTF::clearAll()
{
    VkDevice dv = creator->_device;

    descriptorPool.destroy_pools(dv);
    creator->destroy_buffer(materialDataBuffer);

    for (auto &[k, v] : meshes)
    {
        creator->destroy_buffer(v->meshBuffers.indexBuffer);
        creator->destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto &[k, v] : images)
    {
        if (v.image == creator->_errorCheckerBoardImage.image)
        {
            continue;
        }
        creator->destroy_image(v);
    }

    for (auto &sampler : samplers)
    {
        vkDestroySampler(dv, sampler, nullptr);
    }
}
std::optional<AllocatedImage> load_image(VulkanEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image)
{
    AllocatedImage newImage{};
    int width, height, nrChannels;

    std::visit(fastgltf::visitor{
                   [](auto &arg) {},
                   [&](fastgltf::sources::URI &filePath) {
                       assert(filePath.fileByteOffset == 0);
                       assert(filePath.uri.isLocalPath());

                       const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                       unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                       if (data)
                       {
                           VkExtent3D imageSize;
                           imageSize.width = width;
                           imageSize.height = height;
                           imageSize.depth = 1;

                           newImage = engine->create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM,
                                                           VK_IMAGE_USAGE_SAMPLED_BIT, false);

                           stbi_image_free(data);
                       }
                   },
                   [&](fastgltf::sources::Vector &vector) {
                       unsigned char *data = stbi_load_from_memory(
                           vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);

                       if (data)
                       {
                           VkExtent3D imageSize;
                           imageSize.width = width;
                           imageSize.height = height;
                           imageSize.depth = 1;

                           newImage = engine->create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM,
                                                           VK_IMAGE_USAGE_SAMPLED_BIT, false);

                           stbi_image_free(data);
                       }
                   },
                   [&](fastgltf::sources::BufferView &view) {
                       auto &bufferView = asset.bufferViews[view.bufferViewIndex];
                       auto &buffer = asset.buffers[bufferView.bufferIndex];

                       std::visit(fastgltf::visitor{
                                      [](auto &arg) {},
                                      [&](fastgltf::sources::Vector &vector) {
                                          unsigned char *data = stbi_load_from_memory(
                                              vector.bytes.data() + bufferView.byteOffset,
                                              static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, 4);
                                          if (data)
                                          {
                                              VkExtent3D imageSize;
                                              imageSize.width = width;
                                              imageSize.height = height;
                                              imageSize.depth = 1;

                                              newImage = engine->create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM,
                                                                              VK_IMAGE_USAGE_SAMPLED_BIT, false);

                                              stbi_image_free(data);
                                          }
                                      }},
                                  buffer.data);
                   },
               },
               image.data);

    if (newImage.image == VK_NULL_HANDLE)
    {
        return {};
    }
    else
    {
        return newImage;
    }
}
