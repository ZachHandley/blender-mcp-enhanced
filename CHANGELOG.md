# Changelog

All notable changes to the blender-mcp project will be documented in this file.

## [Enhanced] - 2025-01-17

### Added - MCP Enhancement: Persistent Context & Tool Chaining + Geometry Nodes Support

#### 🔄 **Shared Context Storage**
- **Persistent Variables**: Variables now persist between `execute_blender_code` calls
  - Access via `shared['variable_name']` in scripts
  - Survives across multiple tool executions
  - Automatic cleanup (keeps last 50 operations in history)

- **Object & Material Handles**: Reference Blender objects/materials across tool calls
  - `get_object('handle_name')` - retrieve stored object references
  - `get_material('handle_name')` - retrieve stored material references
  - `store_object('handle', 'obj_name')` - store object by handle
  - `store_material('handle', 'mat_name')` - store material by handle

- **Operation History**: Track recent operations for debugging
  - Automatic logging of all tool executions
  - Timestamp and result tracking
  - Configurable history length (default: 50 operations)

#### 🛠️ **New MCP Tools**
Added 7 new tools for better workflow composition:

1. **`get_shared_context`** - Inspect current shared state
   - Shows all persistent variables, handles, and history count
   - Useful for debugging and understanding current context

2. **`clear_shared_context`** - Reset context selectively
   - Clear all context or specific sections (variables, objects, materials, operations, history)
   - Fresh start when needed

3. **`get_operation_history`** - View recent operations
   - Configurable count (default: 10 recent operations)
   - Shows operation type, input, result, and timestamp

4. **`create_object_handle`** - Manually create object handles
   - Map custom handle names to Blender objects
   - Returns object info (type, location) for verification

5. **`create_material_handle`** - Manually create material handles
   - Map custom handle names to Blender materials
   - Returns material info (uses_nodes) for verification

6. **`list_object_handles`** - List all object handles
   - Shows handle → object mapping with details
   - Current location, type, visibility status

7. **`list_material_handles`** - List all material handles
   - Shows handle → material mapping with details
   - Node usage and reference count

#### ⚡ **Enhanced Tool Chaining**
- **Auto-Handle Creation**: Asset downloads now automatically create handles
  - `download_polyhaven_asset` (models) → returns `object_handles` dict
  - `download_polyhaven_asset` (textures) → returns `material_handle` string
  - MCP responses show created handles for easy reference

- **Enhanced Script Execution**: `execute_blender_code` improvements
  - Pre-populated namespace with shared context and helper functions
  - Returns list of current shared variables for debugging
  - Better error handling with history tracking

#### 📁 **Files Modified**
- **`addon.py`**: Added shared context storage, new tool handlers, enhanced returns
- **`src/blender_mcp/server.py`**: Added 7 new MCP tool endpoints, enhanced descriptions

### Technical Details

#### **Shared Context Structure**
```python
self.shared_context = {
    'variables': {},  # User-defined variables (shared['key'] = value)
    'objects': {},    # Object references by handle
    'materials': {},  # Material references by handle
    'operations': {}, # Operation results by ID
    'history': []     # Operation history (last 50)
}
```

#### **Enhanced Script Namespace**
Scripts now have access to:
```python
{
    "bpy": bpy,                                    # Standard Blender API
    "shared": self.shared_context['variables'],    # Persistent variables
    "get_object": lambda handle: ...,              # Retrieve object handles
    "get_material": lambda handle: ...,            # Retrieve material handles
    "store_object": self._store_object_handle,     # Store object handles
    "store_material": self._store_material_handle, # Store material handles
    "store_operation": self._store_operation_result # Store operation results
}
```

#### **Backward Compatibility**
- ✅ All existing functionality preserved
- ✅ Existing scripts continue to work without modification
- ✅ New features are additive only
- ✅ No breaking changes to existing MCP tools

#### **Geometry Nodes Integration**
Enhanced with comprehensive procedural modeling capabilities:

**New MCP Tools:**
- **`complete_geometry_node`** - Create sophisticated geometry node networks for procedural modeling
- **`get_geometry_nodes_status`** - Check Geometry Nodes availability and Blender version compatibility

**Procedural Modeling Features:**
- **AI-Driven Creation**: Claude can create complex parametric objects (tables, chairs, organic shapes)
- **Node Network Construction**: Full geometry node network creation with custom inputs and properties
- **Blender 4.x Support**: Automatic compatibility handling for Blender 3.x and 4.x interface differences
- **Shared Context Integration**: Auto-creates object handles for seamless tool chaining

**Example Usage:**
```python
# Create a procedural table with geometry nodes
complete_geometry_node(
    object_name="ProceduralTable",
    nodes=[
        {"type": "NodeGroupInput", "location": [0, 0]},
        {"type": "GeometryNodeMeshCube", "location": [200, 200], "inputs": {"Size": [2, 0.1, 1]}},  # Table top
        {"type": "GeometryNodeMeshCube", "location": [200, 0], "inputs": {"Size": [0.1, 1.8, 0.1]}},    # Table leg
        {"type": "GeometryNodeJoinGeometry", "location": [400, 100]},
        {"type": "NodeGroupOutput", "location": [600, 100]}
    ],
    links=[
        {"from_node": 1, "from_socket": "Mesh", "to_node": 3, "to_socket": 0},
        {"from_node": 2, "from_socket": "Mesh", "to_node": 3, "to_socket": 0},
        {"from_node": 3, "from_socket": "Geometry", "to_node": 4, "to_socket": "Geometry"}
    ]
)
# Returns: object_handle: "geometry_ProceduralTable" for easy script access

# Use in subsequent operations
execute_blender_code("table = get_object('geometry_ProceduralTable'); table.location.z = 1")
```

### Usage Examples

#### **Persistent Variables**
```python
# Tool call 1: Store data
execute_blender_code("shared['cube_count'] = 5")

# Tool call 2: Use stored data
execute_blender_code("for i in range(shared['cube_count']): bpy.ops.mesh.primitive_cube_add()")
```

#### **Object Handles**
```python
# Create handle manually
create_object_handle("main_cube", "Cube")

# Use handle in script
execute_blender_code("cube = get_object('main_cube'); cube.location.z = 2")

# Auto-handles from downloads
download_polyhaven_asset(asset_id="chair", asset_type="models")
# Returns: object_handles: {"imported_chair_0": "Chair_01"}
execute_blender_code("chair = get_object('imported_chair_0'); chair.scale = (2,2,2)")
```

#### **Material Handles**
```python
# Download texture (auto-creates handle)
download_polyhaven_asset(asset_id="wood_floor", asset_type="textures")
# Returns: material_handle: "material_wood_floor"

# Apply to object
execute_blender_code("mat = get_material('material_wood_floor'); bpy.context.object.data.materials.append(mat)")
```

#### **Context Management**
```python
# Inspect current state
get_shared_context()

# View recent operations
get_operation_history(count=5)

# Start fresh
clear_shared_context()
```

### Benefits

1. **🔄 Multi-Step Workflows**: Build complex scenes across multiple tool calls
2. **🔗 Tool Chaining**: Use results from one tool in subsequent operations
3. **🏷️ Easy References**: Handle system makes object/material management simple
4. **🐛 Better Debugging**: Operation history and context inspection tools
5. **⚡ Improved Productivity**: No more repeating setup code in every script
6. **🔒 Backward Compatible**: Existing workflows continue to work unchanged
7. **🔧 Procedural Modeling**: AI-driven geometry nodes for sophisticated parametric objects
8. **🎯 Blender 4.x Support**: Full compatibility with modern Blender geometry node workflows

---

## Previous Versions

### [Original] - Before 2025-01-17
- Basic MCP server with isolated script execution
- Individual tools for Blender operations
- PolyHaven, Sketchfab, and Hyper3D integrations
- No persistence between tool calls