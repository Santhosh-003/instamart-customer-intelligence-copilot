-- Q1: Find customers whose completed orders above ₹500
-- have a combined value greater than ₹8,000.

SELECT
    customer_id,
    SUM(order_amount) AS total_spend,
    COUNT(*) AS qualifying_orders
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Q2: Calculate revenue for each city and product category.

SELECT
    c.city,
    p.category,
    SUM(o.order_amount) AS total_revenue,
    COUNT(*) AS completed_orders
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, total_revenue DESC;

-- Q3: Find products priced above the average catalogue price_inr.

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;

-- Q4: Find customers who have no completed orders.

SELECT
    c.customer_id,
    c.city,
    c.age,
    c.gender
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Find customers whose completed-order spending
-- is greater than the average spending across all customers.

WITH customer_spending AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend
FROM customer_spending
WHERE total_spend > (
    SELECT AVG(total_spend)
    FROM customer_spending
)
ORDER BY total_spend DESC;

-- Q6: Rank customers by completed-order spending within each city.

WITH customer_spending AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spending_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_spending_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC, customer_id
    ) AS row_number
FROM customer_spending
ORDER BY city, spending_rank, customer_id;

-- Q7: Calculate daily completed-order revenue and running total.

WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_day,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_day,
    daily_revenue,
    SUM(daily_revenue) OVER (
        ORDER BY order_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM daily_revenue
ORDER BY order_day;