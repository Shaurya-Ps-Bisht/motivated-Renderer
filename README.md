# motivated Renderer
 I got motivated, so I decided to make a renderer using Vulkan


## Building
Uses C-make, for Windows generate a visual studio solution.

On linux, generate a single config or multi-config make. On Arch, I use Ninja to generate a multi-config make using these commands

```C-make
cmake -S . -B build -G "Ninja Multi-Config"
cmake --build build --config Debug
 ```
