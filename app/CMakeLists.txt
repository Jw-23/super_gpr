add_executable(project_app src/main.cpp)

set_target_properties(project_app PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

target_link_libraries(
  project_app
  PRIVATE project_core project_options project_warnings
)
