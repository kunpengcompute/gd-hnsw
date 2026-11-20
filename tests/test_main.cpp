#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  const auto pattern=(std::filesystem::temp_directory_path()/"gd-hnsw-ut-XXXXXX").string();
  std::vector<char> buffer(pattern.begin(),pattern.end());
  buffer.push_back('\0');
  const char* temporary=::mkdtemp(buffer.data());
  if (!temporary) { std::cerr << "Unable to create isolated test directory\n"; return 2; }
  const std::filesystem::path isolated(temporary);
  if (::setenv("TMPDIR",temporary,1)!=0) return 2;
  const int result=RUN_ALL_TESTS();
  std::error_code error;
  std::filesystem::remove_all(isolated,error);
  if(error) { std::cerr<<"Temporary directory cleanup: "<<error.message()<<'\n'; return 2; }
  return result;
}
