# 修改记录

## 2026-10-28 生成阶段诊断
按用户授权仅增加阶段日志及with_loops生成入口faulthandler：共享模块导入、原生模块路径、初始化、cell、FR、每100环、数组/几何校验、网络、写data/VTK、finalize。提交记录节点及作业号，python -X faulthandler -u。未调整物理参数或覆盖结果；此为诊断，不是已修复Bus error。

## 2026-10-28 排错
用户截图显示with_loops生成阶段python generate.py收到Bus error，尚无对应stdout或调用栈。核对生成调用包括pyexadis导入、initialize、Cell、ExaDisNet、write_data、write_vtk、finalize；不能据此判定内存不足或物理参数错误。先索要同作业.out及无环组状态，未改模拟代码。

## 2026-10-27
按用户指定rho1e12模板调整：test顶部state及逐项模块定义，增加--init；每组generate入口，提交直接调用本组generate；生成初态VTK和FR布局CSV；父目录plot增加--input及两组缺失检查。
参考子目录没有plot.py，使用其父目录plot.py作为参考。所有本轮新增/修改均在E:/openDIS下，不再写D盘日志。
保持BCC_NL300K、Ngrid128、1um盒子、FR/环参数及弛豫判据不变。未修改参考算例及核心代码。
