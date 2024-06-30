# motivated Renderer
 I got motivated, so I decided to make a renderer using Vulkan


## Building
Uses C-make, for Windows generate a visual studio solution.

On linux, generate a single config or multi-config make. On Arch, I use Ninja to generate a multi-config make using these commands

```C-make
cmake -S . -B build -G "Ninja Multi-Config"
cmake --build build --config Debug
 ```
Build shaders separately to get the spv files, I use:

```C-make
cmake --build build --config Debug --target Shaders
 ```
## Tudu

- [ ] Downgrade clang (17.0.3) in arch to match visual studio, or the other way around
- [x] Venus
- [x] Earth (Orbit/Moon)
- [x] Mars
