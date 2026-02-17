import simpy
import random
import math
import numpy as np
import matplotlib.pyplot as plt

#set constants and parameters like in the doc

SIM_TIME = 8 * 60  # 8 hours in minutes

MEAN_PREP = 9
STD_PREP = 2

LOAD_TIME = 2
UNLOAD_TIME = 2
DRONE_SPEED = 60 / 60  # 60 km/h = 1 km/min
TAKEOFF_TIME = 0.5
LAND_TIME = 0.5

TAKEOFF_CHARGE = 4
LAND_CHARGE = 1
CHARGE_PER_KM = 6
CHARGE_RATE = 4  # % per minute

AREA_LIMIT = 6

ORDER_VALUE_MEAN = 145
ORDER_VALUE_STD = 41

#monitored resource class to track utilization of chefs and drones

class MonitoredResource(simpy.Resource):
    """Resource that tracks utilization"""
    def __init__(self, env, capacity):
        super().__init__(env, capacity)
        self.busy_time = 0
        self.last_changed = 0
        self.env = env

    def request(self, *args, **kwargs):
        self._update_busy()
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        self._update_busy()
        return super().release(*args, **kwargs)

    def _update_busy(self):
        duration = self.env.now - self.last_changed
        self.busy_time += len(self.users) * duration
        self.last_changed = self.env.now

    def utilization(self, sim_time):
        self._update_busy()
        if self.capacity == 0:
            return 0
        return self.busy_time / (sim_time * self.capacity)

#holder for all stats

class Stats:
    def __init__(self):
        self.wait_for_chef = []
        self.wait_for_drone = []
        self.delivery_times = []
        self.order_values = []

#simulation is same order as in doc

def order_process(env, order_id, chefs, drones, stats):
    """Process a single order from arrival to delivery"""
    arrival_time = env.now
    
    # Generate customer location and order value
    x = random.uniform(-AREA_LIMIT, AREA_LIMIT)
    y = random.uniform(-AREA_LIMIT, AREA_LIMIT)
    distance = math.sqrt(x**2 + y**2)
    stats.order_values.append(random.normalvariate(ORDER_VALUE_MEAN, ORDER_VALUE_STD))
    
    # Chef preparation phase
    with chefs.request() as chef_req:
        yield chef_req
        stats.wait_for_chef.append(env.now - arrival_time)
        prep_time = max(0, random.normalvariate(MEAN_PREP, STD_PREP))
        yield env.timeout(prep_time)
    
    ready_time = env.now
    
    # Drone delivery phase
    with drones.request() as drone_req:
        yield drone_req
        stats.wait_for_drone.append(env.now - ready_time)
        
        # Load and fly to customer
        yield env.timeout(LOAD_TIME)
        yield env.timeout(TAKEOFF_TIME)
        yield env.timeout(distance / DRONE_SPEED)
        yield env.timeout(LAND_TIME)
        yield env.timeout(UNLOAD_TIME)
        
        # Delivery complete - record time i did it wrong for a long time and was recording time to return to base instead of delivery time it drove me crazy becuase i could not get less then 30 min delivery times.
        stats.delivery_times.append(env.now - arrival_time)
        
        # Return and recharge (doesn't affect delivery time)
        yield env.timeout(TAKEOFF_TIME)
        yield env.timeout(distance / DRONE_SPEED)
        yield env.timeout(LAND_TIME)
        
        battery_used = 2 * (TAKEOFF_CHARGE + LAND_CHARGE + distance * CHARGE_PER_KM)
        recharge_time = battery_used / CHARGE_RATE
        yield env.timeout(recharge_time)

def order_generator(env, chefs, drones, stats, mean_interarrival):
    """Generate orders with exponential interarrival times"""
    order_id = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / mean_interarrival))
        env.process(order_process(env, order_id, chefs, drones, stats))
        order_id += 1

#run simulation function

def run_simulation(num_chefs, num_drones, mean_interarrival, random_seed=None):
    """Run simulation and return statistics"""
    if random_seed is not None:
        random.seed(random_seed)
    
    env = simpy.Environment()
    chefs = MonitoredResource(env, num_chefs)
    drones = MonitoredResource(env, num_drones)
    stats = Stats()
    
    env.process(order_generator(env, chefs, drones, stats, mean_interarrival))
    env.run(until=SIM_TIME)
    
    return stats, chefs.utilization(SIM_TIME), drones.utilization(SIM_TIME)


def print_stats_summary(data, name):
    """Print summary statistics"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    p50 = np.percentile(data, 50)
    p90 = np.percentile(data, 90)
    
    print(f"\n{name}")
    print(f"  Mean: {mean_val:.2f} | Std: {std_val:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}")
    print(f"  50th: {p50:.2f} | 90th: {p90:.2f}")
    
    return {'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val, 'p50': p50, 'p90': p90}

def plot_all_histograms(stats):
   #2 x2 charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    datasets = [
        (stats.wait_for_chef, "Wait for Chef (min)", axes[0, 0]),
        (stats.wait_for_drone, "Wait for Drone (min)", axes[0, 1]),
        (stats.delivery_times, "Total Delivery Time (min)", axes[1, 0]),
        (stats.order_values, "Order Values ($)", axes[1, 1])
    ]
    
    for data, title, ax in datasets:
        mean_val = np.mean(data)
        p90_val = np.percentile(data, 90)
        
        ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(p90_val, color='orange', linestyle='--', linewidth=2, label=f'90th: {p90_val:.1f}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# optimal chef to drone finder

def find_optimal_config(target=30, mean_ia=7, max_runs=3):
    
    #Find minimum staffing to achieve target delivery time.
    
    print("time under 30 min")
    print(f"{'Chefs':<8} | {'Drones':<8} | {'Avg Delivery':<15} | {'90th Pctl':<12} | ")
    print("-" * 55)
    
    for c in range(2, 8):
        for d in range(2, 30):#this range is large but should not slow down program as it stops when it finds a solution
            all_deliveries = []
            for run in range(max_runs):
                stats, _, _ = run_simulation(c, d, mean_ia, random_seed=42+run)
                all_deliveries.extend(stats.delivery_times)
            
            if not all_deliveries:
                continue
            
            avg_delivery = np.mean(all_deliveries)
            p90 = np.percentile(all_deliveries, 90)
            
           
            if avg_delivery <= target:
                print(f"{c:<8} | {d:<8} | {avg_delivery:<15.2f} | {p90:<12.2f} ")
                return c, d, avg_delivery, p90
            elif d % 5 == 0:  
                print(f"{c:<8} | {d:<8} | {avg_delivery:<15.2f} | {p90:<12.2f} ")
    
    return None

# main exicution block

if __name__ == "__main__":
    
    # ===== PART A =====

    print("PART A: BASELINE SIMULATION (2 chefs, 2 drones, 7 min/order)")
    
    
    stats, chef_util, drone_util = run_simulation(2, 2, 7, random_seed=42)
    
    print(f"\nOrders Completed: {len(stats.delivery_times)}")
    print(f"Chef Utilization: {chef_util:.1%} | Drone Utilization: {drone_util:.1%}")
    
    # Print statistics
    print_stats_summary(stats.wait_for_chef, "Wait Time for Chef (minutes)")
    print_stats_summary(stats.wait_for_drone, "Wait Time for Drone (minutes)")
    delivery_stats = print_stats_summary(stats.delivery_times, "Total Delivery Time (minutes)")
    print_stats_summary(stats.order_values, "Order Values ($)")
    
 #all charts in one for ease of download
    plot_all_histograms(stats)
    
    # ===== PART B - QUESTION 1 =====
    print("PART B - QUESTION 1: 90th Percentile Delivery Time")
    print(f"\nAnswer: {delivery_stats['p90']:.2f} minutes")
    
    # ===== PART B - QUESTION 2 =====
    print("PART B - QUESTION 2: Optimal Staffing for Avg Delivery ≤ 30 min")
       
    result = find_optimal_config(target=30, mean_ia=7, max_runs=3)
    
    if result:
        c, d, avg, p90 = result
        print(f"Answer: {c} chefs and {d} drones")
        print(f"Average Delivery Time: {avg:.2f} minutes")
        print(f"90th Percentile: {p90:.2f} minutes")
    else:
        print("\nNo configuration found within search range. or probably need to expand search limits.")