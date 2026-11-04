// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/threading.h"

#include "colmap/util/logging.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

namespace {

// Custom implementation of std::barrier that allows us to execute the below
// tests deterministically.
class Barrier {
 public:
  Barrier() : Barrier(2) {}

  explicit Barrier(const size_t count)
      : threshold_(count), count_(count), generation_(0) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto current_generation = generation_;
    if (!--count_) {
      ++generation_;
      count_ = threshold_;
      condition_.notify_all();
    } else {
      condition_.wait(lock, [this, current_generation] {
        return current_generation != generation_;
      });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  const size_t threshold_;
  size_t count_;
  size_t generation_;
};

}  // namespace

// IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
//            so we use glog's CHECK macros inside threads.

TEST(Thread, Wait) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, WaitRethrowsException) {
  class TestThread : public Thread {
   public:
    void Run() override { throw std::runtime_error("Error"); }
  };

  TestThread thread;
  thread.Start();
  EXPECT_THROW(thread.Wait(), std::runtime_error);
  EXPECT_TRUE(thread.IsFinished());
  // The exception is only rethrown once.
  EXPECT_NO_THROW(thread.Wait());
  // A thread that fails before signaling its setup does not block the caller.
  thread.Start();
  EXPECT_FALSE(thread.CheckValidSetup());
  EXPECT_THROW(thread.Wait(), std::runtime_error);
}

TEST(Thread, Pause) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier pauseBarrier;
    Barrier pausedBarrier;
    Barrier resumedBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      pauseBarrier.Wait();
      pausedBarrier.Wait();
      BlockIfPaused();
      resumedBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.pauseBarrier.Wait();
  thread.Pause();
  thread.pausedBarrier.Wait();
  while (!thread.IsPaused() || thread.IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_TRUE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Resume();
  thread.resumedBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, PauseStop) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier pauseBarrier;
    Barrier pausedBarrier;
    Barrier resumedBarrier;
    Barrier stopBarrier;
    Barrier stoppingBarrier;
    Barrier stoppedBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      pauseBarrier.Wait();
      pausedBarrier.Wait();
      BlockIfPaused();
      resumedBarrier.Wait();
      stopBarrier.Wait();
      stoppingBarrier.Wait();

      if (IsStopped()) {
        stoppedBarrier.Wait();
        endBarrier.Wait();
        return;
      }
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.pauseBarrier.Wait();
  thread.Pause();
  thread.pausedBarrier.Wait();
  while (!thread.IsPaused() || thread.IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_TRUE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Resume();
  thread.resumedBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.stopBarrier.Wait();
  thread.Stop();
  thread.stoppingBarrier.Wait();
  thread.stoppedBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_TRUE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_TRUE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, Restart) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  for (size_t i = 0; i < 2; ++i) {
    thread.Start();

    thread.startBarrier.Wait();
    EXPECT_TRUE(thread.IsStarted());
    EXPECT_FALSE(thread.IsStopped());
    EXPECT_FALSE(thread.IsPaused());
    EXPECT_TRUE(thread.IsRunning());
    EXPECT_FALSE(thread.IsFinished());

    thread.endBarrier.Wait();
    thread.Wait();
    EXPECT_TRUE(thread.IsStarted());
    EXPECT_FALSE(thread.IsStopped());
    EXPECT_FALSE(thread.IsPaused());
    EXPECT_FALSE(thread.IsRunning());
    EXPECT_TRUE(thread.IsFinished());
  }
}

TEST(Thread, ValidSetup) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier signalBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      SignalValidSetup();
      signalBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.signalBarrier.Wait();
  EXPECT_TRUE(thread.CheckValidSetup());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, InvalidSetup) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier signalBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      SignalInvalidSetup();
      signalBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.signalBarrier.Wait();
  EXPECT_FALSE(thread.CheckValidSetup());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, Callback) {
  class TestThread : public Thread {
   public:
    enum Callbacks {
      CALLBACK1,
      CALLBACK2,
    };

    TestThread() {
      RegisterCallback(CALLBACK1);
      RegisterCallback(CALLBACK2);
    }

   private:
    void Run() {
      Callback(CALLBACK1);
      Callback(CALLBACK2);
    }
  };

  bool called_back1 = false;
  std::function<void()> CallbackFunc1 = [&called_back1]() {
    called_back1 = true;
  };

  bool called_back2 = false;
  std::function<void()> CallbackFunc2 = [&called_back2]() {
    called_back2 = true;
  };

  bool called_back3 = false;
  std::function<void()> CallbackFunc3 = [&called_back3]() {
    called_back3 = true;
  };

  TestThread thread;
  thread.AddCallback(TestThread::CALLBACK1, CallbackFunc1);
  thread.Start();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_FALSE(called_back2);
  EXPECT_FALSE(called_back3);

  called_back1 = false;
  called_back2 = false;
  thread.AddCallback(TestThread::CALLBACK2, CallbackFunc2);
  thread.Start();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_TRUE(called_back2);
  EXPECT_FALSE(called_back3);

  called_back1 = false;
  called_back2 = false;
  called_back3 = false;
  thread.AddCallback(TestThread::CALLBACK1, CallbackFunc3);
  thread.Start();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_TRUE(called_back2);
  EXPECT_TRUE(called_back3);
}

TEST(Thread, DefaultCallback) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier signalBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      endBarrier.Wait();
    }
  };

  bool called_back1 = false;
  std::function<void()> CallbackFunc1 = [&called_back1]() {
    called_back1 = true;
  };

  bool called_back2 = false;
  std::function<void()> CallbackFunc2 = [&called_back2]() {
    called_back2 = true;
  };

  TestThread thread;
  thread.AddCallback(TestThread::STARTED_CALLBACK, CallbackFunc1);
  thread.AddCallback(TestThread::FINISHED_CALLBACK, CallbackFunc2);
  thread.Start();
  thread.startBarrier.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_FALSE(called_back2);
  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_TRUE(called_back2);
}

TEST(Thread, LifecycleFailuresPreserveCompletionNotifications) {
  class TestThread : public Thread {
   public:
    bool fail_run = false;
    bool ran = false;
    void Run() override {
      ran = true;
      SignalValidSetup();
      if (fail_run) throw std::runtime_error("run failure");
    }
  };
  for (const int phase : {0, 1, 2, 3}) {
    SCOPED_TRACE(phase);
    TestThread thread;
    thread.fail_run = phase == 1;
    bool notified = false;
    int succeeded = 0;
    thread.AddCallback(Thread::STARTED_CALLBACK, [&] {
      if (phase == 0) throw std::runtime_error("started failure");
    });
    thread.AddCallback(Thread::SUCCEEDED_CALLBACK, [&] {
      ++succeeded;
      if (phase == 2) throw std::runtime_error("success failure");
    });
    thread.AddCallback(Thread::FINISHED_CALLBACK,
                       [] { throw std::runtime_error("finished failure"); });
    thread.AddCallback(Thread::FINISHED_CALLBACK, [&] { notified = true; });
    thread.AddCallback(Thread::FINISHED_CALLBACK,
                       [] { throw std::runtime_error("later failure"); });
    thread.Start();
    try {
      thread.Wait();
      FAIL() << "Expected lifecycle failure";
    } catch (const std::runtime_error& error) {
      const char* messages[] = {"started failure",
                                "run failure",
                                "success failure",
                                "finished failure"};
      EXPECT_STREQ(error.what(), messages[phase]);
    }
    EXPECT_EQ(succeeded, phase >= 2 ? 1 : 0);
    EXPECT_TRUE(notified);
    EXPECT_TRUE(thread.IsFinished());
    EXPECT_EQ(thread.ran, phase != 0);
    EXPECT_EQ(thread.CheckValidSetup(), phase != 0);
  }
}

TEST(Thread, SuccessCallbackSkipsStoppedRun) {
  class TestThread : public Thread {
   public:
    bool stop_run = false;
    void Run() override {
      SignalValidSetup();
      if (stop_run) Stop();
    }
  };
  for (const bool stopped : {false, true}) {
    SCOPED_TRACE(stopped);
    TestThread thread;
    thread.stop_run = stopped;
    int succeeded = 0, finished = 0;
    thread.AddCallback(Thread::SUCCEEDED_CALLBACK, [&] { ++succeeded; });
    thread.AddCallback(Thread::FINISHED_CALLBACK, [&] { ++finished; });
    thread.Start();
    EXPECT_NO_THROW(thread.Wait());
    EXPECT_EQ(succeeded, stopped ? 0 : 1);
    EXPECT_EQ(finished, 1);
  }
}

TEST(Thread, CompletionFailureAllowsRestart) {
  class TestThread : public Thread {
    void Run() override { SignalValidSetup(); }
  } thread;
  bool fail_once = true;
  int completions = 0;
  thread.AddCallback(Thread::FINISHED_CALLBACK, [&] {
    if (fail_once) {
      fail_once = false;
      throw std::runtime_error("completion failure");
    }
  });
  thread.AddCallback(Thread::FINISHED_CALLBACK, [&] { ++completions; });
  thread.Start();
  EXPECT_THROW(thread.Wait(), std::runtime_error);
  thread.Start();
  EXPECT_NO_THROW(thread.Wait());
  EXPECT_EQ(completions, 2);
}

TEST(Thread, RestartJoinsThrowingCompletionWithoutHoldingStateMutex) {
  class TestThread : public Thread {
    void Run() override { SignalValidSetup(); }
  } thread;
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto released = release.get_future().share();
  int callbacks = 0;
  thread.AddCallback(Thread::FINISHED_CALLBACK, [&] {
    if (++callbacks == 1) {
      entered.set_value();
      released.wait();
      throw std::runtime_error("old completion failure");
    }
  });
  thread.Start();
  const auto entered_status = entered_future.wait_for(std::chrono::seconds(5));
  if (entered_status != std::future_status::ready) {
    release.set_value();
    EXPECT_THROW(thread.Wait(), std::runtime_error);
    ASSERT_EQ(entered_status, std::future_status::ready);
  }
  EXPECT_TRUE(thread.IsFinished());
  auto restart = std::async(std::launch::async, [&] { thread.Start(); });
  EXPECT_EQ(restart.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  release.set_value();
  ASSERT_EQ(restart.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_NO_THROW(restart.get());
  EXPECT_NO_THROW(thread.Wait());
  EXPECT_EQ(callbacks, 2);
}

}  // namespace

TEST(Thread, StopDuringRestartIsNotLost) {
  class TestThread : public Thread {
   public:
    std::atomic<int> runs{0};
    std::atomic<bool> second_run_stopped{false};
    void Run() override {
      SignalValidSetup();
      if (++runs == 2) second_run_stopped = IsStopped();
    }
  } thread;
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto released = release.get_future().share();
  int completions = 0;
  int successes = 0;
  thread.AddCallback(Thread::SUCCEEDED_CALLBACK, [&] { ++successes; });
  thread.AddCallback(Thread::FINISHED_CALLBACK, [&] {
    if (++completions == 1) {
      entered.set_value();
      released.wait();
    }
  });
  thread.Start();
  const auto entered_status = entered_future.wait_for(std::chrono::seconds(5));
  if (entered_status != std::future_status::ready) {
    release.set_value();
    thread.Wait();
    ASSERT_EQ(entered_status, std::future_status::ready);
  }
  auto restart = std::async(std::launch::async, [&] { thread.Start(); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool starting = false;
  while (!starting && std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(thread.mutex_);
      starting = thread.starting_;
    }
    if (!starting) std::this_thread::yield();
  }
  if (starting) thread.Stop();
  release.set_value();
  ASSERT_EQ(restart.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_NO_THROW(restart.get());
  EXPECT_NO_THROW(thread.Wait());
  ASSERT_TRUE(starting);
  EXPECT_EQ(thread.runs.load(), 2);
  EXPECT_TRUE(thread.second_run_stopped.load());
  EXPECT_EQ(successes, 1);
  EXPECT_EQ(completions, 2);
}

namespace {

TEST(ThreadPool, NoArgNoReturn) {
  std::function<void(void)> Func = []() {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  };

  ThreadPool pool(4);
  std::vector<std::shared_future<void>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    future.get();
  }
}

TEST(ThreadPool, ArgNoReturn) {
  std::function<void(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
  };

  ThreadPool pool(4);
  std::vector<std::shared_future<void>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  for (auto& future : futures) {
    future.get();
  }
}

TEST(ThreadPool, NoArgReturn) {
  std::function<int(void)> Func = []() { return 0; };

  ThreadPool pool(4);
  std::vector<std::shared_future<int>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    (void)future.get();
  }
}

TEST(ThreadPool, ArgReturn) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::shared_future<int>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  for (auto& future : futures) {
    (void)future.get();
  }
}

TEST(ThreadPool, Stop) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::shared_future<int>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  pool.Stop();

  EXPECT_THROW(pool.AddTask(Func, 100), std::runtime_error);

  pool.Stop();
}

TEST(ThreadPool, Wait) {
  std::vector<uint8_t> results(100, 0);
  std::function<void(int)> Func = [&results](const int num) {
    results[num] = 1;
  };

  ThreadPool pool(4);
  pool.Wait();

  for (size_t i = 0; i < results.size(); ++i) {
    pool.AddTask(Func, i);
  }

  pool.Wait();

  for (const auto result : results) {
    EXPECT_EQ(result, 1);
  }
}

TEST(ThreadPool, WaitEverytime) {
  std::vector<uint8_t> results(4, 0);
  std::function<void(int)> Func = [&results](const int num) {
    results[num] = 1;
  };

  ThreadPool pool(4);

  for (size_t i = 0; i < results.size(); ++i) {
    pool.AddTask(Func, i);
    pool.Wait();

    for (size_t j = 0; j < results.size(); ++j) {
      if (j <= i) {
        EXPECT_EQ(results[j], 1);
      } else {
        EXPECT_EQ(results[j], 0);
      }
    }
  }

  pool.Wait();
}

TEST(ThreadPool, GetThreadIndex) {
  ThreadPool pool(4);

  std::vector<int> results(100, -1);
  std::function<void(int)> Func = [&](const int num) {
    results[num] = pool.GetThreadIndex();
  };

  for (size_t i = 0; i < results.size(); ++i) {
    pool.AddTask(Func, i);
  }

  pool.Wait();

  for (const auto result : results) {
    EXPECT_GE(result, 0);
    EXPECT_LE(result, 3);
  }
}

TEST(ThreadPool, FuturePropagatesException) {
  ThreadPool pool(1);
  auto future = pool.AddTask([]() { throw std::runtime_error("Error"); });
  EXPECT_THROW(future.get(), std::runtime_error);
  EXPECT_THROW(pool.Wait(), AggregateException);
}

TEST(ThreadPool, WaitPropagatesException) {
  ThreadPool pool(1);
  pool.AddTask([]() { throw std::runtime_error("Error"); });
  EXPECT_THROW(pool.Wait(), AggregateException);
  EXPECT_NO_THROW(pool.Wait());
}

TEST(ThreadPool, WaitPropagatesExceptionWithReturnValue) {
  ThreadPool pool(1);
  pool.AddTask([]() -> int { throw std::runtime_error("Error"); });
  EXPECT_THROW(pool.Wait(), AggregateException);
  EXPECT_NO_THROW(pool.Wait());
}

TEST(ThreadPool, WaitPropagatesMultipleExceptions) {
  ThreadPool pool(1);
  pool.AddTask([]() { throw std::runtime_error("Error 1"); });
  pool.AddTask([]() { throw std::runtime_error("Error 2"); });

  try {
    pool.Wait();
    FAIL() << "Expected AggregateException";
  } catch (const AggregateException& e) {
    EXPECT_EQ(e.exceptions().size(), 2);
  }
  EXPECT_NO_THROW(pool.Wait());
}

TEST(ThreadPool, StopPropagatesException) {
  ThreadPool pool(1);
  pool.AddTask([]() { throw std::runtime_error("Error"); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_THROW(pool.Stop(), AggregateException);
  EXPECT_NO_THROW(pool.Stop());
}

// Stop must report both already published and in-flight task failures only
// after the active worker has finished and queued tasks have been discarded.
class ThreadPoolStopMatrix
    : public testing::TestWithParam<std::pair<bool, bool>> {};

TEST_P(ThreadPoolStopMatrix, JoinsAndConsumesFailures) {
  const bool completed_fails = GetParam().first;
  const bool active_fails = GetParam().second;
  std::promise<void> marker, active_started, release;
  auto release_future = release.get_future().share();
  std::atomic<bool> active_finished{false};
  std::atomic<bool> pending_ran{false};
  ThreadPool pool(1);

  pool.AddTask([completed_fails]() {
    if (completed_fails) {
      throw std::runtime_error("completed failure");
    }
  });
  pool.AddTask([&marker]() { marker.set_value(); });
  marker.get_future().wait();
  pool.AddTask([&]() {
    active_started.set_value();
    release_future.wait();
    active_finished = true;
    if (active_fails) {
      throw std::runtime_error("active failure");
    }
  });
  active_started.get_future().wait();
  auto pending = pool.AddTask([&pending_ran]() { pending_ran = true; });

  std::exception_ptr stop_error;
  bool finished_at_stop = false;
  std::thread stopper([&]() {
    try {
      pool.Stop();
    } catch (...) {
      stop_error = std::current_exception();
    }
    finished_at_stop = active_finished.load();
  });
  const auto status = pending.wait_for(std::chrono::seconds(5));
  release.set_value();
  stopper.join();
  if (status != std::future_status::ready) {
    for (int i = 0; i < 2; ++i) {
      try {
        pool.Stop();
      } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Keep defective-control cleanup from replacing the assertions below.
      }
    }
  }

  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(finished_at_stop);
  EXPECT_FALSE(pending_ran);
  try {
    pending.get();
    ADD_FAILURE() << "Expected broken promise";
  } catch (const std::future_error& e) {
    EXPECT_EQ(e.code(), std::make_error_code(std::future_errc::broken_promise));
  }
  const int expected = static_cast<int>(completed_fails) +
                       static_cast<int>(active_fails);
  if (stop_error) {
    try {
      std::rethrow_exception(stop_error);
    } catch (const AggregateException& e) {
      EXPECT_EQ(e.exceptions().size(), expected);
      const std::string message = e.what();
      EXPECT_EQ(message.find("completed failure") != std::string::npos,
                completed_fails);
      EXPECT_EQ(message.find("active failure") != std::string::npos,
                active_fails);
    } catch (...) {
      ADD_FAILURE() << "Stop returned a non-aggregate error";
    }
  } else {
    EXPECT_EQ(expected, 0);
  }
  EXPECT_NO_THROW(pool.Stop());
  EXPECT_NO_THROW(pool.Wait());
}

INSTANTIATE_TEST_SUITE_P(
    AllOutcomes,
    ThreadPoolStopMatrix,
    testing::Values(std::make_pair(false, false),
                    std::make_pair(true, false),
                    std::make_pair(false, true),
                    std::make_pair(true, true)));

TEST(ThreadPool, FutureAndWaitPropagatesException) {
  ThreadPool pool(1);
  auto future = pool.AddTask([]() { throw std::runtime_error("Error"); });
  EXPECT_THROW(future.get(), std::runtime_error);
  EXPECT_THROW(pool.Wait(), AggregateException);
  EXPECT_NO_THROW(pool.Wait());
}

TEST(ThreadPool, CatchAggregateException) {
  ThreadPool pool(1);
  pool.AddTask([]() { throw std::runtime_error("Error 1"); });
  pool.AddTask([]() { throw std::runtime_error("Error 2"); });

  bool caught = false;
  try {
    pool.Wait();
  } catch (const AggregateException& e) {
    caught = true;
    EXPECT_EQ(e.exceptions().size(), 2);
  }
  EXPECT_TRUE(caught);
}

TEST(ThreadPool, AggregateExceptionMessage) {
  ThreadPool pool(1);
  pool.AddTask([]() { throw std::runtime_error("First error message"); });
  pool.AddTask([]() { throw std::runtime_error("Second error message"); });

  try {
    pool.Wait();
    FAIL() << "Expected AggregateException";
  } catch (const AggregateException& e) {
    std::string what = e.what();
    // Order of exceptions is non-deterministic, so check components separately.
    EXPECT_TRUE(what.find("2 task(s) threw exception(s):") == 0);
    EXPECT_TRUE(what.find("First error message") != std::string::npos);
    EXPECT_TRUE(what.find("Second error message") != std::string::npos);
  }
}

TEST(JobQueue, SingleProducerSingleConsumer) {
  JobQueue<int> job_queue;

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    CHECK_LE(job_queue.Size(), 10);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread.join();
  consumer_thread.join();
}

TEST(JobQueue, SingleProducerSingleConsumerMaxNumJobs) {
  JobQueue<int> job_queue(2);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    CHECK_LE(job_queue.Size(), 2);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread.join();
  consumer_thread.join();
}

TEST(JobQueue, MultipleProducerSingleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread1([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread producer_thread2([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 20; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread1.join();
  producer_thread2.join();
  consumer_thread.join();
}

TEST(JobQueue, SingleProducerMultipleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 20; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread1([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 20);
    }
  });

  std::thread consumer_thread2([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 20);
    }
  });

  producer_thread.join();
  consumer_thread1.join();
  consumer_thread2.join();
}

TEST(JobQueue, MultipleProducerMultipleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread1([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread producer_thread2([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread1([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  std::thread consumer_thread2([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread1.join();
  producer_thread2.join();
  consumer_thread1.join();
  consumer_thread2.join();
}

TEST(JobQueue, Wait) {
  JobQueue<int> job_queue;

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  for (int i = 0; i < 10; ++i) {
    CHECK(job_queue.Push(i));
  }

  std::thread consumer_thread([&job_queue]() {
    CHECK_EQ(job_queue.Size(), 10);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_EQ(job.Data(), i);
    }
  });

  job_queue.Wait();

  EXPECT_EQ(job_queue.Size(), 0);
  EXPECT_TRUE(job_queue.Push(0));
  EXPECT_TRUE(job_queue.Pop().IsValid());

  consumer_thread.join();
}

TEST(JobQueue, StopProducer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  Barrier stopBarrier;
  std::thread producer_thread([&job_queue, &stopBarrier]() {
    CHECK(job_queue.Push(0));
    stopBarrier.Wait();
    CHECK(!job_queue.Push(0));
  });

  stopBarrier.Wait();
  EXPECT_EQ(job_queue.Size(), 1);

  job_queue.Stop();
  producer_thread.join();

  EXPECT_FALSE(job_queue.Push(0));
  EXPECT_FALSE(job_queue.Pop().IsValid());
}

TEST(JobQueue, StopConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  EXPECT_TRUE(job_queue.Push(0));

  Barrier pop_barrier;
  std::thread consumer_thread([&job_queue, &pop_barrier]() {
    const auto job = job_queue.Pop();
    CHECK(job.IsValid());
    CHECK_EQ(job.Data(), 0);
    pop_barrier.Wait();
    CHECK(!job_queue.Pop().IsValid());
  });

  pop_barrier.Wait();
  EXPECT_EQ(job_queue.Size(), 0);

  job_queue.Stop();
  consumer_thread.join();

  EXPECT_FALSE(job_queue.Push(0));
  EXPECT_FALSE(job_queue.Pop().IsValid());
}

TEST(JobQueue, Clear) {
  JobQueue<int> job_queue(1);

  EXPECT_TRUE(job_queue.Push(0));
  EXPECT_EQ(job_queue.Size(), 1);

  EXPECT_EQ(job_queue.Clear(), 1);
  EXPECT_EQ(job_queue.Size(), 0);
  EXPECT_EQ(job_queue.Clear(), 0);
}

TEST(JobQueue, ClearDestroysJobsWithConcurrentProducerAndWaiter) {
  JobQueue<std::shared_ptr<int>> job_queue(1);
  auto value = std::make_shared<int>(7);
  std::weak_ptr<int> weak_value = value;
  ASSERT_TRUE(job_queue.Push(value));
  value.reset();

  Barrier producer_barrier;
  Barrier waiter_barrier;
  std::thread producer([&] {
    producer_barrier.Wait();
    CHECK(job_queue.Push(std::make_shared<int>(8)));
  });
  std::thread waiter([&] {
    waiter_barrier.Wait();
    job_queue.Wait();
  });
  producer_barrier.Wait();
  waiter_barrier.Wait();
  EXPECT_EQ(job_queue.Clear(), 1);
  EXPECT_TRUE(weak_value.expired());
  producer.join();
  // The producer may refill before the waiter observes the empty queue.
  EXPECT_EQ(job_queue.Clear(), 1);
  waiter.join();
  EXPECT_EQ(job_queue.Size(), 0);
}

TEST(GetEffectiveNumThreads, Nominal) {
  EXPECT_GT(GetEffectiveNumThreads(-2), 0);
  EXPECT_GT(GetEffectiveNumThreads(-1), 0);
  EXPECT_GT(GetEffectiveNumThreads(0), 0);
  EXPECT_EQ(GetEffectiveNumThreads(1), 1);
  EXPECT_EQ(GetEffectiveNumThreads(2), 2);
  EXPECT_EQ(GetEffectiveNumThreads(3), 3);
}

}  // namespace
}  // namespace colmap
