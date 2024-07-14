#pragma once

#include "VkBootstrap.h"
#include "vk_loader.h"
#include <cstdint>
#include <functional>
#include <vk_descriptors.h>
#include <vk_initializers.h>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

constexpr unsigned int FRAME_OVERLAP = 2;

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect
{
    const char *name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

struct DeletionQueue
{
    std::deque<std::function<void()>> deletors; // a dequeue of functions returning void, no params

    void push_function(std::function<void()> &&function)
    { // rvalue reference
        deletors.push_back(function);
    }

    void flush()
    {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
        {
            (*it)(); // derefernce the iterator to obtain the function and using () to exec it
        }
        deletors.clear();
    }
};

struct FrameData
{
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkSemaphore _swapchainSemaphore, _renderSemaphore;
    VkFence _renderFence;

    DeletionQueue _deletionQueue;
};

class VulkanEngine
{

  public:
    FrameData _frames[FRAME_OVERLAP];
    FrameData &get_current_frame()
    {
        return _frames[_frameNumber % FRAME_OVERLAP];
    }; // alternate between frame 1 and 2, double buffer

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{1700, 900};

    struct SDL_Window *_window{nullptr};
    static VulkanEngine &Get();

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;
    VkSurfaceKHR _surface;

    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;

    DescriptorAllocator globalDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    void init();
    void cleanup();
    void draw();
    void run();

    void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);
    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

  private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_default_data();
    void init_sync_structures();
    void draw_background(VkCommandBuffer cmd);
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void resize_swapchain();

    void init_descriptors();
    void init_pipelines();
    void init_background_pipelines();
    void init_imgui();
    void init_triangle_pipeline();
    void init_mesh_pipeline();
    void draw_geometry(VkCommandBuffer cmd);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer &buffer);

    GPUMeshBuffers rectangle;

    DeletionQueue _mainDeletionQueue;
    VkExtent2D _drawExtent;
    float renderScale = 1.f;

    VmaAllocator _allocator;
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;

    VkPipelineLayout _trianglePipelineLayout;
    VkPipelineLayout _meshPipelineLayout;

    VkPipeline _meshPipeline;
    VkPipeline _trianglePipeline;

    std::vector<ComputeEffect> backgroundEffects;
    std::vector<std::shared_ptr<MeshAsset>> testMeshes;
    int currentBackgroundEffect{0};

    bool resize_request;
};
